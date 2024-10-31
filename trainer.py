import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm.auto import tqdm
import wandb

from utils import compute_f1
import time

def get_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )


def fp16_zo_eval(func):
    def wrapper(self, *args, **kwargs):
        zo_eval_dtype = torch.float16
        dtype = self.model.dtype
        if self.args.optimizer == "zo" and dtype != torch.float16:
            # self.model.to(zo_eval_dtype)
            # print("Switched to fp16 for evaluation")
            copy_learnable_params = dict()
            print(f"Max memory 0: {torch.cuda.max_memory_allocated() / 1024**3}")
            for name, param in self.trainable_parameters.items():
                copy_learnable_params[name] = param.clone().detach()
            print(f"Max memory 1: {torch.cuda.max_memory_allocated() / 1024**3}")
            self.model.to(zo_eval_dtype)
            # Make sure the model is in fp16
            for name, param in self.trainable_parameters.items():
                assert param.dtype == torch.float16, f"Param {name} is not in fp16"
            print("Switched to fp16 for evaluation")
        
        print(f"Max memory 2: {torch.cuda.max_memory_allocated() / 1024**3}")
        # Clear memory
        torch.cuda.empty_cache()
        result = func(self, *args, **kwargs)
        print(f"Max memory 3: {torch.cuda.max_memory_allocated() / 1024**3}")

        if self.args.optimizer == "zo" and dtype != torch.float16:
        #     self.model.to(dtype)
        #     print("Switched back to original dtype")
            for name, param in self.trainable_parameters.items():
                param.data = copy_learnable_params[name].data
            self.model.to(dtype)
            # Make sure the model is back to original dtype
            for name, param in self.trainable_parameters.items():
                assert param.dtype == dtype, f"Param {name} is not in original dtype"
            print("Switched back to original dtype")    
        
        return result
    return wrapper



class Trainer:
    def __init__(self, args, model, tokenizer, train_dataloader, eval_dataloader, accelerator, cls_idx=None, optimizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.trainable_parameters = dict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.trainable_parameters[name] = param
        
        get_trainable_parameters(model)

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criteria = nn.CrossEntropyLoss()

        if self.args.optimizer == "fo":
            assert args.n == 1, "Only n=1 is supported for first-order optimization"
            self.one_step = self.one_step_fo
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
            self.optimizer = optimizer
        else:
            if self.args.zo_mode == "single":
                self.one_step = self.one_step_zo_single
            else:
                self.one_step = self.one_step_zo_dual
        
        if self.args.task_name == "copa" or self.args.task_name == "winogrande":
            self.eval = self.eval_mch
        elif self.args.task_name == "squad" or self.args.task_name == "drop":
            self.eval = self.eval_qa
        else:
            self.eval = self.eval_cls
            self.cls_idx0 = cls_idx[0]
            self.cls_idx1 = cls_idx[1]

    def train(self):
        progress_bar = tqdm(range(self.args.num_train_epochs * len(self.train_dataloader)))
        self.global_step = 0
        self.max_accuracy = 0
        for self.epoch in range(self.args.num_train_epochs):
            for batch in self.train_dataloader:
                self.rand_seed = torch.randint(0, 10000000, (1,)).item()
                loss = self.one_step(batch)

                if (self.global_step + 1) % self.args.logging_steps == 0:
                    self.eval()

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})

                wandb.log({"train/loss": loss.item(), "epoch": self.epoch, "step": self.global_step})
                self.global_step += 1

                if self.global_step >= self.args.max_iterations:
                    return


    def one_step_fo(self, batch):
        self.model.train()
        outputs = self.model(**batch)
        loss = outputs.loss
        # loss.backward()
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss


    def one_step_zo_single(self, batch):
        self.model.train()
        with torch.no_grad():
            if self.args.split:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
            else:
                input_ids = batch['input_ids'].repeat(self.args.n, 1)
                attention_mask = batch['attention_mask'].repeat(self.args.n, 1)
                labels = batch['labels'].repeat(self.args.n, 1)

            torch.manual_seed(self.rand_seed)
            for name, param in self.trainable_parameters.items():
                z = self.args.eps * torch.randn_like(param)
                param.data.add_(z)

            logits = self.model(input_ids, attention_mask, return_dict=False)[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            del logits

            # Compute loss for each n
            shift_logits = shift_logits.view(self.args.n, -1, shift_logits.size(-1))
            shift_labels = shift_labels.view(self.args.n, -1)

            loss1 = torch.stack([self.criteria(shift_logits[i], shift_labels[i]) for i in range(self.args.n)])

            del shift_logits, shift_labels
            
            torch.manual_seed(self.rand_seed)
            for name, param in self.trainable_parameters.items():
                z = -2 * self.args.eps * torch.randn_like(param)
                param.data.add_(z)

            logits = self.model(input_ids, attention_mask, return_dict=False)[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            del logits
            # Compute loss for each n
            shift_logits = shift_logits.view(self.args.n, -1, shift_logits.size(-1))
            shift_labels = shift_labels.view(self.args.n, -1)

            loss2 = torch.stack([self.criteria(shift_logits[i], shift_labels[i]) for i in range(self.args.n)])
            del shift_logits, shift_labels

            torch.manual_seed(self.rand_seed)
            for name, param in self.trainable_parameters.items():
                z = self.args.eps * torch.randn_like(param)
                param.data.add_(z)

            projected_grad = (loss1 - loss2) / (2.0 * self.args.eps)

            torch.manual_seed(self.rand_seed)
            for name, param in self.trainable_parameters.items():
                if self.args.n == 1:
                    z = -self.args.learning_rate * projected_grad * torch.randn_like(param)
                else:
                    z = -self.args.learning_rate * (projected_grad.view(-1, 1, 1) * torch.randn_like(param)).mean(dim=0, keepdim=True)
                param.data.add_(z)

        return loss1.mean()
        
    def one_step_zo_dual(self, batch):
        self.model.train()
        with torch.no_grad():
            input_ids = batch['input_ids'].repeat(2*self.args.n, 1)
            attention_mask = batch['attention_mask'].repeat(2*self.args.n, 1)
            labels = batch['labels'].repeat(2*self.args.n, 1)

            torch.manual_seed(self.rand_seed)
            for name, param in self.trainable_parameters.items():
                z = self.args.eps * torch.randn_like(param.data[:self.args.n])
                param.data[:self.args.n].add_(z)
                param.data[self.args.n:].sub_(z)

            logits = self.model(input_ids, attention_mask, return_dict=False)[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            del logits

            # Compute loss for each n
            shift_logits = shift_logits.view(2, self.args.n, -1, shift_logits.size(-1))
            shift_labels = shift_labels.view(2, self.args.n, -1)

            loss1 = torch.stack([self.criteria(shift_logits[0, i], shift_labels[0, i]) for i in range(self.args.n)])
            loss2 = torch.stack([self.criteria(shift_logits[1, i], shift_labels[1, i]) for i in range(self.args.n)])

            del shift_logits, shift_labels

            torch.manual_seed(self.rand_seed)
            for name, param in self.trainable_parameters.items():
                z = self.args.eps * torch.randn_like(param.data[:self.args.n])
                param.data[:self.args.n].sub_(z)
                param.data[self.args.n:].add_(z)
            
            projected_grad = (loss1 - loss2) / (2.0 * self.args.eps)

            torch.manual_seed(self.rand_seed)
            for name, param in self.trainable_parameters.items():
                if self.args.n == 1:
                    z = self.args.learning_rate * torch.randn_like(param.data[:self.args.n])
                else:
                    z = self.args.learning_rate * (projected_grad.view(-1, 1, 1) * torch.randn_like(param.data[:self.args.n])).mean(dim=0, keepdim=True)
                param.data[:self.args.n].sub_(z)
                param.data[self.args.n:].sub_(z)

        return loss1.mean()

    @fp16_zo_eval
    def eval_cls(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for step, batch in enumerate(self.eval_dataloader): 
                labels = batch["input_ids"][:, -1]
                batch.pop("labels")
                outputs = self.model(**batch)

                logits = outputs.logits

                neg_logits = logits[:, -2, self.cls_idx0]
                pos_logits = logits[:, -2, self.cls_idx1]
                predictions = torch.where(neg_logits > pos_logits, self.cls_idx0, self.cls_idx1)

                correct += (predictions == labels).sum().item()
                total += len(labels)

            accuracy = correct / total
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy

            print(f"step {self.global_step} accuracy: {accuracy}")
            wandb.log({"eval/accuracy": accuracy ,"epoch": self.epoch, "step": self.global_step, "eval/max_accuracy": self.max_accuracy})

    @fp16_zo_eval
    def eval_mch(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for step, batch in enumerate(self.eval_dataloader):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                outputs = self.model(input_ids, attention_mask=attention_mask)

                logits = outputs.logits

                log_probs = F.log_softmax(logits, dim=-1)  # Shape: [bsz, seq_len, vocab_size]

                # iterate over the batch every two examples
                for i in range(0, len(log_probs), 2):
                    valid_len1 = (labels[i] != -100).sum()
                    valid_len2 = (labels[i + 1] != -100).sum()

                    valid_log_probs1 = log_probs[i, -(valid_len1+1):-1]
                    valid_log_probs2 = log_probs[i + 1, -(valid_len2+1):-1]

                    valid_log_probs1 = valid_log_probs1[range(len(valid_log_probs1)), labels[i, -valid_len1:]]
                    valid_log_probs2 = valid_log_probs2[range(len(valid_log_probs2)), labels[i + 1, -valid_len2:]]

                    valid_log_probs1 = valid_log_probs1.mean()
                    valid_log_probs2 = valid_log_probs2.mean()

                    correct += (valid_log_probs1 > valid_log_probs2).item()
                    total += 1
            
            accuracy = correct / total
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy

            print(f"step {self.global_step} accuracy: {accuracy}")
            wandb.log({"eval/accuracy": accuracy ,"epoch": self.epoch, "step": self.global_step, "eval/max_accuracy": self.max_accuracy})

    @fp16_zo_eval    
    def eval_qa(self):
        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            all_f1_scores = []
            samples = 0
            for step, batch in enumerate(self.eval_dataloader):
                for i in range(len(batch['input_ids'])):
                    input_ids = batch['input_ids'][i].unsqueeze(0)
                    labels = batch['labels'][i]

                    valid_len = (labels != -100).sum()
                    valid_labels = labels[-valid_len:]

                    valid_input_ids = input_ids[:, :-valid_len]

                    outputs = self.model.generate(
                        valid_input_ids, do_sample=False, temperature=1.0, max_new_tokens=50,
                        num_beams=1, top_p=0.95, top_k=None, num_return_sequences=1, 
                        eos_token_id=[self.tokenizer.encode("\n", add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
                    )

                    outputs = outputs[0][valid_input_ids.size(1):]

                    # Convert tensors to strings for F1 computation
                    pred_str = self.tokenizer.decode(outputs, skip_special_tokens=True)
                    label_str = self.tokenizer.decode(valid_labels, skip_special_tokens=True)

                    # print(pred_str, "#######", label_str)
                    all_f1_scores.append(compute_f1([pred_str], [label_str]))
                    samples += 1
                print(f"sample {samples}, avg f1: {np.mean(all_f1_scores)}")
            print("Evaluation time: ", time.time() - start_time)
            avg_f1 = np.mean(all_f1_scores)
            if avg_f1 > self.max_accuracy:
                self.max_accuracy = avg_f1

            print(f"step {self.global_step} f1: {avg_f1}")
            wandb.log({"eval/f1": avg_f1, "epoch": self.epoch, "step": self.global_step, "eval/max_f1": self.max_accuracy})
            # Exit if the f1 score is 1.0
            if avg_f1 < 0.1:
                exit()