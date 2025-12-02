from tqdm                                       import tqdm
from torchmetrics.functional                    import accuracy, f1_score, precision
import torch.nn                                 as nn
import numpy                                    as np
import torch
import sys
import re


class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


class BaseObject(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        else:
            return self._name


class Accuracy(BaseObject):
    __name__ = 'accuracy'

    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        self.classes = classes
        self.m       = nn.Softmax(dim=1)

    def forward(self, prediction, y):
        y = torch.argmax(y, dim=1)
        prediction = torch.argmax(self.m(prediction), dim=1)
        acc = accuracy(prediction, y, 'multiclass', num_classes=self.classes)
        return acc.float().mean().item()


class Fscore(BaseObject):
    def __init__(self, classes, targets, **kwargs):
        super().__init__(**kwargs)
        self.classes = classes
        self.targets = targets
        self.m       = nn.Softmax(dim=1)
        self._get_name()

    def _get_name(self):
        name_dict  = {0:'normal', 1:'gist', 2:'leiomyoma', 3:'schwannoma'}
        class_name = name_dict[self.targets]
        self._name = f'f_{class_name}'

    def forward(self, prediction, y):
        y = torch.argmax(y, dim=1)
        prediction = torch.argmax(self.m(prediction), dim=1)
        f1  = f1_score(prediction, y, 'multiclass', num_classes=self.classes, average=None)
        return f1[self.targets].float().mean().item()


class WeightedPrecision(BaseObject):
    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        self.classes = classes
        self.m       = nn.Softmax(dim=1)
        self._name   = 'weighted_precision' 

    def forward(self, prediction, y):
        # one-hot → label
        y = torch.argmax(y, dim=1)
        prediction = torch.argmax(self.m(prediction), dim=1)
        prec = precision(
            prediction,
            y,
            task='multiclass',
            num_classes=self.classes,
            average='weighted'
        )
        return prec.float().mean().item()


class WeightedF1(BaseObject):
    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        self.classes = classes
        self.m       = nn.Softmax(dim=1)
        self._name   = 'weighted_f1' 

    def forward(self, prediction, y):
        # one-hot → label
        y = torch.argmax(y, dim=1)
        prediction = torch.argmax(self.m(prediction), dim=1)
        f1 = f1_score(
            prediction,
            y,
            task='multiclass',
            num_classes=self.classes,
            average='weighted'
        )
        return f1.float().mean().item()



class EarlyStopping:
    '''
    stop training if the validation loss dose not imporved within the patience
    '''
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Epoch:
    def __init__(self, model, classes, loss, metrics, stage_name, verbose=True):
        self.model = model
        self.classes = classes
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)


    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'loss': loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y) 
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, classes, loss, metrics, optimizer, scheduler, verbose=True):
        super().__init__(
            model=model,
            classes = classes,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            verbose=verbose,
        )
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = optimizer
        self.scheduler = scheduler

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        if self.classes == 1:
            y = y.float().view(-1, 1)
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, classes, loss, metrics, verbose=True):
        super().__init__(
            model=model,
            classes = classes,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            verbose=verbose,
        )
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        if self.classes == 1:
            y = y.float().view(-1, 1)
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
    
class EvaluationEpoch(Epoch):
    def __init__(self, model, classes, loss, metrics, verbose=True):
        super().__init__(
            model=model,
            classes=classes,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            verbose=verbose,
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def on_epoch_start(self):
        self.model.eval()
        # Completely freeze BN layers
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
                module.track_running_stats = False

    def batch_update(self, x, y):
        if self.classes == 1:
            y = y.float().view(-1, 1)
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction

    def run(self, dataloader):
        self.on_epoch_start()
        
        # Store results on CPU (memory saving)
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0
        
        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not self.verbose,
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)
                
                # Immediately move from GPU to CPU to save memory
                all_predictions.append(y_pred.cpu())
                all_targets.append(y.cpu())
                
                # Accumulate loss
                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Clean up GPU memory
                del x, y, y_pred, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Concatenate all data (on CPU)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        logs = {}
        logs['loss'] = total_loss / total_samples
        
        # Move to GPU for metric calculation (if needed)
        device_for_metrics = self.device if all_predictions.numel() < 1000000 else 'cpu'  # 100만개 미만이면 GPU 사용
        
        all_predictions = all_predictions.to(device_for_metrics)
        all_targets = all_targets.to(device_for_metrics)
        
        for metric_fn in self.metrics:
            # Move metric function to same device
            metric_fn.to(device_for_metrics)
            metric_value = metric_fn(all_predictions, all_targets)
            logs[metric_fn.__name__] = metric_value
        
        if self.verbose:
            s = self._format_logs(logs)
            print(f"\n{self.stage_name} - {s}")
        
        return logs
    
    def run_memory_efficient(self, dataloader, chunk_size=10000):
        """Calculate metrics in chunks to minimize memory usage"""
        self.on_epoch_start()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc=self.stage_name):
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)
                
                all_predictions.append(y_pred.cpu())
                all_targets.append(y.cpu())
                
                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Process intermediate chunks when reaching chunk size
                current_size = sum(pred.size(0) for pred in all_predictions)
                if current_size >= chunk_size:
                    # Process intermediate results and free memory
                    chunk_preds = torch.cat(all_predictions, dim=0)
                    chunk_targets = torch.cat(all_targets, dim=0)
                    
                    all_predictions.clear()
                    all_targets.clear()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Process remaining data
        if all_predictions:
            final_preds = torch.cat(all_predictions, dim=0)
            final_targets = torch.cat(all_targets, dim=0)
            
            # Calculate final metrics
            logs = {'loss': total_loss / total_samples}
            for metric_fn in self.metrics:
                metric_value = metric_fn(final_preds, final_targets)
                logs[metric_fn.__name__] = metric_value
            
            return logs