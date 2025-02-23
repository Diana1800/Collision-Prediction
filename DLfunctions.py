from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import time
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gradient_and_lr(gradient_norms, learning_rates, epoch):
    # Create a figure with two subplots
    plt.figure(figsize=(14, 5))

    # Plot the Gradient Norms
    plt.subplot(1, 2, 1)
    plt.plot(gradient_norms, label='Gradient Norms')
    plt.title(f'Gradient Norms for Epoch {epoch + 1}')
    plt.xlabel('Batch')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(True)

    # Plot the Learning Rates
    plt.subplot(1, 2, 2)
    plt.plot(learning_rates, label='Learning Rate', color='orange')
    plt.title(f'Learning Rate for Epoch {epoch + 1}')
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'gradient_lr_epoch_{epoch + 1}.png')  # Save plot
    plt.close()  # Close figure to avoid overlapping issues in next epoch

def plot_confusion_matrix(y_true, y_pred, epoch, is_binary=True):
    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    
    if is_binary:
        plt.title(f'Confusion Matrix for Epoch {epoch + 1} (Binary)')
        plt.xticks(ticks=[0.5, 1.5], labels=["Class 0", "Class 1"])
        plt.yticks(ticks=[0.5, 1.5], labels=["Class 0", "Class 1"], rotation=0)
    else:
        plt.title(f'Confusion Matrix for Epoch {epoch + 1} (Multiclass)')
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_epoch_{epoch + 1}.png')  # Save confusion matrix plot
    plt.close()

def plot_loss(history, epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.title(f'Loss for Epoch {epoch + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_epoch_{epoch + 1}.png')
    plt.close()

def plot_prob_distribution(probs, epoch):
    plt.figure(figsize=(10, 6))
    sns.histplot(probs, bins=30, kde=True, color='purple')
    plt.title(f'Distribution of Predicted Probabilities for Epoch {epoch + 1}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'prob_distribution_epoch_{epoch + 1}.png')
    plt.close()


def Train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=10, device='cuda', is_binary=True, save_metric='f1', l1_lambda=None, oTBLogger=None):
    history = {'train_loss': [], 'val_loss': [], 'train_score': [], 'val_score': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
    'val_precision': [], 'val_recall': [], 'val_f1': [], 'epoch_time': [], 'learning_rate': [], 'last_saved_epoch': None, 'batch_learning_rate': [], 'gradient_norms': []}
    
    best_val_loss = float('inf')
    best_val_f1 = 0.0  # Assuming F1 score ranges between 0 and 1
    last_saved_epoch_loss = 0
    last_saved_epoch_f1 = 0
    total_norm = 0
    
    model = model.to(device)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # ======= Training Phase =======
        model.train()  # training mode
        train_losses = []
        all_train_labels = []
        all_train_preds = []
        current_epoch_gradient_norms = []  # Track gradient norms for the current epoch
        current_epoch_lrs = []  # Track learning rates for the current epoch
        all_train_probs = [] #To Plot the distribution of predicted probabilities

        
        total_train_batches = len(train_loader)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            total_norm = 0  # Reset for each batch
            if isinstance(images, list) or isinstance(images, tuple):
                # If multiple images (e.g Siamese network), move each to the device
                images = [img.to(device) for img in images]
                outputs = model(*images)  # Unpack and pass them as separate arguments
            else:
                # If a single image tensor, move it to the device
                images = images.to(device)
                outputs = model(images)  # Forward pass

            labels = labels.to(device)

            optimizer.zero_grad()

            loss = criterion(outputs, labels)  # Compute loss

            # Optional L1 Regularization
            if l1_lambda is not None and l1_lambda > 0:
                l1_norm = sum(param.abs().sum() for param in model.parameters())
                loss += l1_lambda * l1_norm

            loss.backward()  # Backward pass
            for p in model.parameters():  # Calculate gradient norm
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            current_epoch_gradient_norms.append(total_norm)
            history['gradient_norms'].append(total_norm)
            optimizer.step()  # Update weights

            if scheduler is not None:
                scheduler.step()  
                current_lr = optimizer.param_groups[0]['lr']
                current_epoch_lrs.append(current_lr)
                history['batch_learning_rate'].append(current_lr)


            train_losses.append(loss.item())

            # Generate predictions for each batch based on if Binary or Multiclass
            if is_binary:
                probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
                preds = torch.round(torch.sigmoid(outputs)).detach().cpu().numpy().flatten()
            else:
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy().flatten()
                probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                probs = probs[:, 1]  # For class 1 probability in binary

            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().detach().numpy().flatten().astype(int))
            all_train_probs.extend(probs)

            # Print batch iteration info
            print(f'\rTrain Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{total_train_batches} - Loss: {loss.item():.4f} | Gradient Norm: {total_norm:.4f} | Allocated: {(torch.cuda.memory_allocated() / (1024 ** 2)):.2f} MB | Reserved: {(torch.cuda.memory_reserved() / (1024 ** 2)):.2f} MB', end='')
        print('', end = '\r')

        train_loss = sum(train_losses) / len(train_losses)
        train_score = accuracy_score(all_train_labels, all_train_preds)  

        average_method = 'binary' if is_binary else 'macro'
        train_precision = precision_score(all_train_labels, all_train_preds, average=average_method, zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, average=average_method, zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, average=average_method, zero_division=0)

        # Plot and save gradient and learning rate changes
        plot_gradient_and_lr(current_epoch_gradient_norms, current_epoch_lrs, epoch)
        plot_prob_distribution(all_train_probs, epoch)


        # ======= Validation Phase =======
        model.eval()
        val_losses = []
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []
        
        total_val_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                if isinstance(images, list) or isinstance(images, tuple):
                    # If multiple images (e.g Siamese network), move each to the device
                    images = [img.to(device) for img in images]
                    outputs = model(*images)  # Unpack and pass them as separate arguments
                else:
                    # If a single image tensor, move it to the device
                    images = images.to(device)
                    outputs = model(images)  # Forward pass

                labels = labels.to(device)

                loss = criterion(outputs, labels)  # Compute loss
                val_losses.append(loss.item())

                # Generate predictions for each batch based on binary or Multiclass
                if is_binary:
                    preds = torch.round(torch.sigmoid(outputs)).detach().cpu().numpy().flatten()
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                else:
                    # _, preds = torch.max(outputs, 1)
                    # preds = preds.cpu().numpy().flatten()
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                    probs = probs[:, 1]  # For class 1 probability in binary

                all_val_preds.extend(preds)
                all_val_labels.extend(labels.cpu().numpy().flatten().astype(int))
                all_val_probs.extend(probs)

                # Print batch iteration info
                print(f'\rValidation Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{total_val_batches} - Loss: {loss.item():.4f}- Gradient Norm: {total_norm:.4f}', end='')
            print('', end = '\r')           # Clear the last line after the iteration ends    
        
        val_loss = sum(val_losses) / len(val_losses)
        val_score = accuracy_score(all_val_labels, all_val_preds)  # Validation accuracy
        average_method = 'binary' if is_binary else 'macro'  
        val_precision = precision_score(all_val_labels, all_val_preds, average=average_method, zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, average=average_method, zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, average=average_method, zero_division=0)

        # Plot and save confusion matrix and loss
        plot_loss(history, epoch)
        plot_confusion_matrix(all_val_labels, all_val_preds, epoch, is_binary)

        # ======= Scheduler Step evry epoch =======
        # For schedulers like StepLR or ExponentialLR that adjust the learning rate based on the epoch.
        # if scheduler is not None: 
        #     scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start_time


        # ====== Saving & Prints =======
        # Logging for history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_score'].append(train_score)
        history['val_score'].append(val_score)
        history['epoch_time'].append(epoch_time)
        history['learning_rate'].append(current_lr)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)


        print(f"Epoch {epoch + 1}| Train Loss: {train_loss:.4f}      | Val Loss: {val_loss:.4f}      | Train Score: {train_score:.4f}  | Val Score: \033[1;33m{val_score:.4f}\033[0m  | Learning Rate: {current_lr:.6f}, Time: {epoch_time:.2f}s | ", flush=True)
        print(f'\nEpoch {epoch + 1}| Train Precision: {train_precision:.4f} | Val Precision: \033[1m{val_precision:.4f}\033[0m | Train Recall: {train_recall:.4f} | Val Recall: \033[1m{val_recall:.4f}\033[0m | F1 Train Score: {train_f1:.4f} | F1 Val Score: {val_f1:.4f}', flush=True)

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts_loss = copy.deepcopy(model.state_dict())
            last_saved_epoch_loss = epoch + 1
            history['last_saved_epoch_loss'] = last_saved_epoch_loss

            save_path_loss = f'Best_Model_{epoch + 1}.pth'   # {epoch + 1} - add to name to save every epoch
            save_dict_loss = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'metric': 'loss'
            }
            if scheduler is not None:
                save_dict_loss['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(save_dict_loss, save_path_loss)
            print(f"--> Saved as best model based on loss at epoch {epoch + 1}", flush=True)

        # Check if the current model has the highest validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts_f1 = copy.deepcopy(model.state_dict())
            last_saved_epoch_f1 = epoch + 1
            history['last_saved_epoch_f1'] = last_saved_epoch_f1

            save_path_f1 = f'Best_Model_{epoch + 1}.pth'  # Save path for the best F1 model
            save_dict_f1 = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_f1': best_val_f1,
                'metric': 'f1'
            }
            if scheduler is not None:
                save_dict_f1['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(save_dict_f1, save_path_f1)
            print(f"--> Saved as best model based on F1 score at epoch {epoch + 1}", flush=True)


    if save_metric == 'f1':
        model.load_state_dict(best_model_wts_f1)
        print(f"Best model saved based on F1 score at epoch {last_saved_epoch_f1}.")
    else:
        model.load_state_dict(best_model_wts_loss)
        print(f"Best model saved based on loss at epoch {last_saved_epoch_loss}.")

    
    return model, history
 
 
 
