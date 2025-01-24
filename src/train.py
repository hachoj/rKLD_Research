
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import math
import os

from config import config
from data_loader import DataLoaderLite
from model import SLM
from lr_schedular import get_lr
from hellaswag import *

if __name__ == '__main__':
    model_name = "SLM-0.124B_random_testing"

    # ----------------------------------------------------------------------
    # Setting up DDP
    # torchrun --standalone --nproc_per_node=NUMGPUS train.py

    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run
    print(f"using ddp: {ddp}")
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to the rank
        assert torch.cuda.is_available()
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt  to autodetect device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Running on {device}")

    # ----------------------------------------------------------------------

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    # calculate the number of gradient accumulation steps
    # for the desired batch size
    total_batch_size = 524288
    B=4
    T=1024
    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)


    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', is_rope=config.pos_embd_type == 'ROPE')
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val', is_rope=config.pos_embd_type == 'ROPE')

    torch.set_float32_matmul_precision('high')

    model = SLM(config)
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model    

    # taken from GPT-3 paper
    '''
    TRAINING PARAMS
    '''
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073

    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    # create the log directory we will write checkpoints to and log to
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{model_name}_log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

    if master_process:
        print(f"----------------------------------------")
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
        print(f"ddp_world_size: {ddp_world_size}")
        print(f"max_steps: {max_steps}")
        print(f"----------------------------------------")

    # training loop
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # Model evaluation
        # --------------------------------------------------------------------------------
        # once in a while evaluate our validation loss
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y, pos = val_loader.next_batch()
                    if pos is not None:
                        pos = pos.to(device)
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y, pos)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}") # type: ignore
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")  # type: ignore
                if step > 0 and (step % 5000 == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{model_name}_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()  # type: ignore
                    }
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you wanted to more exactly resume training
                    torch.save(checkpoint, checkpoint_path)
                
        # HellaSwag evaluation
        if (step % 250 == 0 or last_step):
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")
        # --------------------------------------------------------------------------------

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0  # this is just for printing the loss
        for microstep in range(grad_accum_steps):
            x, y, pos = train_loader.next_batch()
            if pos is not None:
                pos = pos.to(device)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y, pos)
            # each loss.backward() call accumulates gradients
            loss = loss / grad_accum_steps # scale the loss for the gradient accumulation
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (microstep == grad_accum_steps - 1)  # type: ignore
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # vairable learning rate
        lr = get_lr(step, warmup_steps, max_steps, min_lr, max_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")  # type: ignore
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")  # type: ignore
    if master_process:
        log_dir = "logs"
        checkpoint_path = os.path.join(log_dir, f"model_{model_name}.pt")
        torch.save(model.state_dict(), checkpoint_path)
    if ddp:
        destroy_process_group()
