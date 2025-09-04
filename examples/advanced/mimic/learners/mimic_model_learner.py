import os
from typing import Union
import pandas as pd
import numpy as np
import tensorflow as tf
from nvflare.apis.fl_constant import FLMetaKey, ReturnCode
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_opt.tf.fedprox_loss import TFFedProxLoss
from mimic.networks.mimic_nets import FCN, get_opt, get_metrics
from sklearn.model_selection import train_test_split
from nvflare.app_common.app_constant import AppConstants, ModelName
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(42)

class MimicModelLearner(ModelLearner):
    def __init__(
        self,
        train_idx_root: str = "./dataset",
        aggregation_epochs: int = 1,
        lr: float = 1e-2,
        fedproxloss_mu: float = 0.0,
        central: bool = False,
        analytic_sender_id: str = "analytic_sender",
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        """Simple tabular data Trainer.

        Args:
            train_idx_root: directory with site training indices for tabular data.
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            lr: local learning rate. Float number. Defaults to 1e-2.
            fedproxloss_mu: weight for FedProx loss. Float number. Defaults to 0.0 (no FedProx).
            central: Bool. Whether to simulate central training. Default False.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            batch_size: batch size for training and validation.
            num_workers: number of workers for data loaders.

        Returns:
            an FLModel with the updated local model differences after running `train()`, the metrics after `validate()`,
            or the best local model depending on the specified task.
        """
        super().__init__()
        self.train_idx_root = train_idx_root
        self.aggregation_epochs = aggregation_epochs
        self.lr = lr
        self.fedproxloss_mu = fedproxloss_mu
        self.best_auc = 0.0
        self.central = central
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.analytic_sender_id = analytic_sender_id

        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0

        self.local_model_file = None
        self.best_local_model_file = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.criterion_prox = None
        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None

    def initialize(self):
        """Initialization of model, optimizer, loss function, etc."""
        self.info(f"Client {self.site_name} initialized at \n {self.app_root} \n with args: {self.args}")

        self.local_model_file = os.path.join(self.app_root, "local_model.weights.h5")
        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.weights.h5")

        self.model = FCN()
        self.optimizer = get_opt()
        self.criterion = tf.keras.losses.BinaryCrossentropy()

        if self.fedproxloss_mu > 0:
            self.info(f"using FedProx loss with mu {self.fedproxloss_mu}")
            self.criterion_prox = TFFedProxLoss(mu=self.fedproxloss_mu)

    def _create_datasets(self):
        """Load the tabular datasets, split for training and validation."""
        if self.train_dataset is None or self.train_loader is None:
            csv_file_path = os.path.join(self.train_idx_root, self.site_name + ".csv")

            if not os.path.exists(csv_file_path):
                self.stop_task(f"No dataset found! File {csv_file_path} does not exist!")

            df = pd.read_csv(csv_file_path)

            # Assumiamo che l'ultima colonna sia il target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            # Divisione train/validation
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = np.array(X_train, dtype=np.float32)
            y_train = np.array(y_train, dtype=np.float32)
            X_valid = np.array(X_valid, dtype=np.float32)
            y_valid = np.array(y_valid, dtype=np.float32)

            self.train_dataset = (X_train, y_train)
            self.valid_dataset = (X_valid, y_valid)

            self.train_loader = tf.data.Dataset.from_tensor_slices(self.train_dataset).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            self.valid_loader = tf.data.Dataset.from_tensor_slices(self.valid_dataset).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def finalize(self):
        """Finalize resources like threads or open files."""
        pass

    def local_train(self, train_loader, global_weights, val_freq: int = 0):
        """Training loop using TensorFlow."""
        for epoch in range(self.aggregation_epochs):
            self.model.trainable = True
            self.epoch_global = self.epoch_of_start_time + epoch
            self.info(f"Local epoch {self.site_name}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})")
            avg_loss = 0.0

            for x, y in train_loader:
                with tf.GradientTape() as tape:
                    out = self.model(x, training=True)
                    base_loss = self.criterion(y, out)
                    if self.fedproxloss_mu > 0:
                        prox = self.criterion_prox(self.model.trainable_variables, list(global_weights.values()))
                        loss = base_loss + prox
                    else:
                        loss = base_loss

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                avg_loss += loss.numpy()

            if val_freq > 0 and epoch % val_freq == 0:
                auc = self.local_valid(self.valid_loader)
                if auc > self.best_auc:
                    self.best_auc = auc
                    self.save_model(is_best=True)

    def save_model(self, is_best=False):
        """Save model weights in HDF5 format"""
        file_path = self.best_local_model_file if is_best else self.local_model_file
        self.model.save_weights(file_path)
        self.model.save_weights(os.path.join(self.app_root, f"{self.site_name}.weights.h5"))

    def train(self, model: FLModel) -> Union[str, FLModel]:
        self._create_datasets()

        # Get round information
        self.info(f"Current/Total Round: {self.current_round + 1}/{self.total_rounds}")
        self.info(f"Client identity: {self.site_name}")

        # Update local model weights with received weights
        global_weights = model.params
        self.model.set_weights(list(global_weights.values()) if isinstance(global_weights, dict) else global_weights)

        # Local train steps
        self.local_train(self.train_loader, global_weights, val_freq=1 if self.central else 0)

        # Validate after local train
        auc = self.local_valid(self.valid_loader)
        self.info(f"val_auc_local_model: {auc:.4f}")

        self.save_model(is_best=False)
        if auc > self.best_auc:
            self.best_auc = auc
            self.save_model(is_best=True)

        # Compute delta model
        local_weights = self.model.get_weights()
        model_dict = {}
        for idx, (layer_name, _) in enumerate(global_weights.items()):
            model_dict[layer_name] = local_weights[idx].astype(np.float32)

        fl_model = FLModel(params_type=ParamsType.FULL, params=model_dict)

        FLModelUtils.set_meta_prop(fl_model, FLMetaKey.NUM_STEPS_CURRENT_ROUND, len(self.train_loader))
        self.info("Local epochs finished. Returning FLModel")
        return fl_model

    def local_valid(self, valid_loader):
        """Validation using TensorFlow."""
        self.model.trainable = False

        auc_metric = get_metrics()[0]
        for inputs, labels in valid_loader:
            outputs = self.model(inputs)

            if outputs.shape[-1] > 1:
                probs = tf.nn.softmax(outputs)[:, 1]  # Per classificazione binaria da softmax
            else:
                probs = tf.nn.sigmoid(outputs)
                probs = tf.squeeze(probs, axis=-1)

            labels = tf.cast(tf.reshape(labels, [-1]), tf.float32)

            auc_metric.update_state(labels, probs)

        return auc_metric.result().numpy()

    def validate(self, model: FLModel) -> Union[str, FLModel]:
        """Validation for global model using TensorFlow."""
        self._create_datasets()

        self.info(f"Client identity: {self.site_name}")

        global_weights = model.params
        self.model.set_weights(list(global_weights.values()) if isinstance(global_weights, dict) else global_weights)

        model_owner = self.get_shareable_header(AppConstants.MODEL_OWNER)

        # perform valid
        train_auc = self.local_valid(self.train_loader)
        self.info(f"AUC: {train_auc:.4f}")

        val_auc = self.local_valid(self.valid_loader)
        self.info(f"AUC {model_owner}: {val_auc:.4f}")
        self.info("Evaluation finished. Returning result")

        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.save_model(is_best=True)

        return FLModel(metrics={"train_auc": float(train_auc), "val_auc": float(val_auc)})

    def get_model(self, model_name: str) -> Union[str, FLModel]:
        if model_name == ModelName.BEST_MODEL:
            try:
                self.model.load_weights(self.best_local_model_file)
                # Get all weights in order (weights + biases for each layer)
                weights = self.model.get_weights()
                # Convert list of weights to dict using sequential keys
                weights_dict = {str(i): weight for i, weight in enumerate(weights)}
                return FLModel(params_type=ParamsType.FULL, params=weights_dict)
            except Exception as e:
                self.error(f"Unable to load best model: {e}")
                return ReturnCode.EXECUTION_RESULT_ERROR
        else:
            raise ValueError(f"Unknown model_type: {model_name}")
