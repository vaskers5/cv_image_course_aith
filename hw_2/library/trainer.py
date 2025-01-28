import torch

from tqdm.auto import tqdm
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader


from library.loss import PerceptualLoss


class Trainer:
    def __init__(self, model, train_loader, optimizer, device, run=None):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.run = run
        self.reconstruction_criterion = MSELoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, leave=False):
            inputs = batch.to(self.device)
            self.optimizer.zero_grad()
            outputs, vq_loss = self.model(inputs)
            reconstruction_loss = self.reconstruction_criterion(outputs, inputs)
            loss = reconstruction_loss + vq_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def train(self, epochs):
        for epoch in tqdm(range(epochs)):
            train_loss, val_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if self.run:
                self.run.track(train_loss, name='train_loss', epoch=epoch)
                self.run.track(val_loss, name='val_loss', epoch=epoch)
                self.run.track(self.optimizer.param_groups[0]['lr'], 
                             name='learning_rate', epoch=epoch)


class AdvancedTrainer:
    def __init__(self, model, train_loader, optimizer, device, val_loader=None, use_perceptual=True, run=None):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.run = run
        self.val_loader = val_loader
        self.reconstruction_criterion = MSELoss()
        
        # Инициализация perceptual loss
        self.perceptual_loss = PerceptualLoss(device) if use_perceptual else None
        
        # Настройка scheduler и чекпоинтов
        self.scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
        self.best_loss = float('inf')
        self.checkpoint_path = 'best_model_checkpoint.pth'
        self.final_checkpoint_path = 'final_model.pth'

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch.to(self.device)
                # outputs, loss = self.model(inputs)
                # optimization - it can not work properly
                outputs, vq_loss = self.model(inputs)
                mse_loss = self.reconstruction_criterion(outputs, inputs)
                perc_loss = self.perceptual_loss(inputs, outputs) if self.perceptual_loss else 0
                loss = mse_loss + vq_loss + 0.1 * perc_loss
                total_loss += loss.item()
        return total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0

    def train_epoch(self, is_fine_tuning=False):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, leave=False):
            inputs = batch.to(self.device)
            self.optimizer.zero_grad()
            # outputs, loss = self.model(inputs)
            # optimization - it can not work properly
            outputs, vq_loss = self.model(inputs)
            mse_loss = self.reconstruction_criterion(outputs, inputs)
            perc_loss = self.perceptual_loss(inputs, outputs) if self.perceptual_loss else 0
            loss = mse_loss + vq_loss + 0.1 * perc_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        
        if is_fine_tuning:
            return total_loss / len(self.train_loader)
        
        avg_train_loss = total_loss / len(self.train_loader)
        val_loss = self.validate_epoch() if self.val_loader else avg_train_loss
        
        self.scheduler.step(val_loss)
        
        if val_loss < self.best_loss:
        # if val_loss > self.best_loss:
            self.best_loss = val_loss
            self._save_checkpoint(avg_train_loss, val_loss)
        
        return avg_train_loss, val_loss

    def _save_checkpoint(self, train_loss, val_loss):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, self.checkpoint_path)
        print(f"\nSaved new best model with val loss {val_loss:.4f}")

    def load_best_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded best checkpoint with val loss {checkpoint['val_loss']:.4f}")

    def _create_full_dataset_loader(self):
        train_dataset = self.train_loader.dataset
        val_dataset = self.val_loader.dataset if self.val_loader else None
        
        if val_dataset:
            full_dataset = ConcatDataset([train_dataset, val_dataset])
            return DataLoader(
                full_dataset,
                batch_size=self.train_loader.batch_size,
                shuffle=True,
                num_workers=self.train_loader.num_workers,
                pin_memory=self.train_loader.pin_memory
            )
        return self.train_loader

    def train(self, epochs, fine_tune_epochs=0):
        for epoch in range(epochs):
            train_loss, val_loss = self.train_epoch()
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if self.run:
                self.run.track(train_loss, name='train_loss', epoch=epoch)
                self.run.track(val_loss, name='val_loss', epoch=epoch)
                self.run.track(self.optimizer.param_groups[0]['lr'], 
                             name='learning_rate', epoch=epoch)

        if fine_tune_epochs > 0:
            print("\nStarting fine-tuning on full dataset")
            self.load_best_checkpoint()
            
            self.train_loader = self._create_full_dataset_loader()
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                
            original_val_loader = self.val_loader
            self.val_loader = None
            self.scheduler = None
            
            for epoch in range(fine_tune_epochs):
                train_loss = self.train_epoch(is_fine_tuning=True)
                print(f"\nFine-tune Epoch {epoch+1}/{fine_tune_epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                
                if self.run:
                    self.run.track(train_loss, name='fine_tune_loss', epoch=epoch)
                    
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.final_checkpoint_path)
            
            self.val_loader = original_val_loader
            print(f"Final model saved to {self.final_checkpoint_path}")
