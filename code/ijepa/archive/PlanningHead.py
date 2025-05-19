import torch
import os, glob, re
from datetime import datetime
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from PIL import Image
import numpy as np

def _auto_resume_path(pattern="checkpoint_epoch*.pth"):
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    epochs = []
    for p in ckpts:
        m = re.search(r"checkpoint_epoch(\d+)\.pth", p)
        if m:
            epochs.append((int(m.group(1)), p))
    return max(epochs, key=lambda x: x[0])[1] if epochs else None

class PlanningHead(nn.Module):
    def __init__(self, ijep_dim, ego_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim*3
        input_dim = ijep_dim + ego_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.output_dim),
        )
        self.num_poses = output_dim

    def forward(self, visual_features, ego_features):
        x = torch.cat([visual_features, ego_features], dim=1)
        flat = self.mlp(x)
        return flat.view(-1, self.num_poses, 3)

    def fit(
        self,
        dataloader,
        ijepa_encoder,
        device,
        epochs: int,
        lr: float,
        optimizer=None,
        criterion=None,
        save_dir: str = ".",
        resume_from: str = None,
        checkpoint_interval: int = 1,
        use_cls_token: bool = False,
        log_interval: int = 100, # Log every N batches + the first one
    ):
        os.makedirs(save_dir, exist_ok=True)
        self.to(device)
        optimizer = AdamW(self.parameters(), lr=lr) if optimizer is None else optimizer
        criterion = nn.L1Loss() if criterion is None else criterion

        start_epoch = 1
        if resume_from is None:
            resume_from = _auto_resume_path(os.path.join(save_dir, "checkpoint_epoch*.pth"))
        if resume_from and os.path.exists(resume_from):
            # (Your existing resume logic here)
            ck = torch.load(resume_from, map_location=device)
            self.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["opt_state"])
            start_epoch = ck["epoch"] + 1
            print(f"Resumed from epoch {ck['epoch']}")


        history = []
        try:
            for epoch in range(start_epoch, epochs + 1):
                self.train()
                total_loss = 0.0
                count = 0
                pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True) # Changed leave=True
                for i, batch in enumerate(pbar): # Use enumerate

                    # --- Check if the entire batch failed collation ---
                    if batch is None:
                        # Optionally print less frequently for None batches
                        if i % log_interval == 0:
                            print(f"\nWarning: Skipping batch {i} because collate_fn returned None.")
                        continue

                    # --- Unpack and check individual tensors ---
                    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                         print(f"\nWarning: Skipping batch {i} due to unexpected batch format: {type(batch)}")
                         continue
                    imgs_raw, ego_raw, gt_raw = batch
                    if imgs_raw is None or ego_raw is None or gt_raw is None:
                        print(f"\nWarning: Skipping batch {i} because it contains None tensors.")
                        continue

                    # Move valid tensors to device
                    imgs = imgs_raw.to(device)
                    ego = ego_raw.to(device)
                    gt = gt_raw.to(device)

                    # --- VERBOSE LOGGING ---
                    # Log first batch (i==0) and every log_interval batches
                    log_this_batch = (i == 0) or ((i + 1) % log_interval == 0)
                    # --- End VERBOSE LOGGING ---

                    with torch.no_grad():
                        out = ijepa_encoder(pixel_values=imgs)
                        feats = None
                        feats = None
                        if use_cls_token and getattr(out, "pooler_output", None) is not None:
                            feats = out.pooler_output
                        elif getattr(out, "last_hidden_state", None) is not None:
                            feats = out.last_hidden_state.mean(1)
                        else:
                            raise ValueError("Model output is missing expected embeddings.")

                        if feats is None:
                             if log_this_batch:
                                 print(f"\nWarning: Feature extraction resulted in None for batch {i}. Skipping.")
                             continue

                    # --- Final check before forward pass ---
                    if feats is None or ego is None:
                         if log_this_batch:
                            print(f"\nError: feats or ego is None before calling PlanningHead forward for batch {i}. Skipping.")
                         continue

                    pred = self(feats, ego) # Pass to PlanningHead.forward
                    loss = criterion(pred, gt)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    current_loss = loss.item()
                    total_loss += current_loss
                    count += 1
                    if count > 0:
                         pbar.set_postfix(loss=f'{total_loss/count:.4f}') # Format loss in progress bar

                    # --- VERBOSE LOGGING BLOCK ---
                    if log_this_batch:
                        print(f"\n--- Logging Batch {i+1}/{len(dataloader)} (Epoch {epoch}) ---")
                        print(f"  Image Batch Shape : {imgs.shape}, Device: {imgs.device}")
                        print(f"  Ego Features Shape: {ego.shape}, Device: {ego.device}")
                        # Print first ego feature vector in the batch
                        print(f"  Ego Features [0]  : {ego[0].cpu().numpy()}")
                        print(f"  IJEPA Feats Shape : {feats.shape}, Device: {feats.device}")
                        # Print first few elements of the first feature vector
                        print(f"  IJEPA Feats [0,:5]: {feats[0, :5].cpu().numpy()}")
                        print(f"  Prediction Shape  : {pred.shape}, Device: {pred.device}")
                        # Print first predicted pose and last predicted pose for the first item in batch
                        print(f"  Prediction [0,0]  : {pred[0, 0].detach().cpu().numpy()}") # Use detach() before cpu()
                        print(f"  Prediction [0,-1] : {pred[0, -1].detach().cpu().numpy()}")
                        print(f"  Ground Truth Shape: {gt.shape}, Device: {gt.device}")
                        # Print first and last ground truth pose for the first item in batch
                        print(f"  Ground Truth [0,0]: {gt[0, 0].cpu().numpy()}")
                        print(f"  Ground Truth [0,-1]: {gt[0, -1].cpu().numpy()}")
                        print(f"  Batch Loss        : {current_loss:.4f}")
                        print(f"---------------------------------------------")
                    # --- End VERBOSE LOGGING BLOCK ---


                # --- Epoch End ---
                avg = total_loss/count if count > 0 else float("nan")
                history.append(avg)
                # Use standard print here so it doesn't get overwritten by tqdm
                print(f"\nEpoch {epoch} finished. Average Loss: {avg:.4f}")

                # (Your existing checkpointing logic here)
                if epoch % checkpoint_interval == 0:
                    ckpt = {
                        "epoch": epoch,
                        "model_state": self.state_dict(),
                        "opt_state": optimizer.state_dict(),
                    }
                    path = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth")
                    torch.save(ckpt, path)
                    print(f"Saved checkpoint: {path}") # Added print confirmation


        except Exception as e: # Capture the exception for logging
            # (Your existing exception handling here)
            print(f"\nTraining interrupted by error at epoch {epoch}, batch {i}: {e}") # Log the error
            # import traceback # Uncomment for full traceback
            # print(traceback.format_exc()) # Uncomment for full traceback
            # Save failure checkpoint
            ckpt = {
                "epoch": epoch,
                "model_state": self.state_dict(),
                "opt_state": optimizer.state_dict(),
            }
            path = os.path.join(save_dir, f"checkpoint_failure_epoch{epoch}.pth")
            torch.save(ckpt, path)
            print(f"Interrupted at epoch {epoch}, saved failure checkpoint {path}")
            raise # Re-raise the exception

        # final save
        # (Your existing final save logic here)
        final_loss = history[-1] if history else float("nan")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Format loss in filename to avoid issues with NaN or characters
        loss_str = f"{final_loss:.4f}".replace('.', '_') if not np.isnan(final_loss) else "NaN"
        fname = f"planning_head_{ts}_loss{loss_str}.pth"
        final_path = os.path.join(save_dir, fname)
        torch.save(self.state_dict(), final_path)
        print(f"Saved final weights to {final_path}")


        # clean old checkpoints
        # (Your existing cleanup logic here)
        cleaned_count = 0
        for ck in glob.glob(os.path.join(save_dir, "checkpoint_epoch*.pth")):
            try:
                os.remove(ck)
                cleaned_count += 1
            except OSError as e:
                print(f"Error removing checkpoint {ck}: {e}")
        if cleaned_count > 0:
            print(f"Cleaned up {cleaned_count} intermediate checkpoints.")


        return history