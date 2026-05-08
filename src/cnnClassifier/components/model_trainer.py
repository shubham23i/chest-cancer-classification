from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path
import tensorflow as tf
import os


class Training:
    def __init__(self, config:TrainingConfig):
        self.config= config

    def get_base_model(self):
        self.model=tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data, "train"),
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode="binary",
            shuffle=True
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data,"valid"),
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode="binary",
            shuffle=False
        )
    def train(self):
        self.steps_per_epoch=self.train_generator.samples//self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
        )
        self.model.save("artifacts/training/model.keras")
        