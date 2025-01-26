import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class AIModelDevelopment:
    def __init__(self, input_shape, num_classes):
        """
        Initialize AI model with specified input shape and number of classes
        
        Args:
            input_shape (tuple): Shape of input data
            num_classes (int): Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def prepare_data(self, X, y, test_size=0.2):
        """
        Prepare and preprocess data for model training
        
        Args:
            X (numpy.array): Input features
            y (numpy.array): Target labels
            test_size (float): Proportion of data for testing
        
        Returns:
            Processed training and testing datasets
        """
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self):
        """
        Create neural network model architecture
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=self.input_shape),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Train the neural network
        
        Args:
            X_train (numpy.array): Training features
            y_train (numpy.array): Training labels
            epochs (int): Number of training iterations
            batch_size (int): Number of samples per gradient update
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (numpy.array): Test features
            y_test (numpy.array): Test labels
        
        Returns:
            Model evaluation metrics
        """
        return self.model.evaluate(X_test, y_test)

# Example usage
def main():
    # Simulated data generation (replace with your actual dataset)
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y = tf.keras.utils.to_categorical(np.random.randint(0, 5, 1000), 5)
    
    # Initialize model
    model_dev = AIModelDevelopment(input_shape=(10,), num_classes=5)
    
    # Prepare data
    X_train, X_test, y_train, y_test = model_dev.prepare_data(X, y)
    
    # Train model
    history = model_dev.train_model(X_train, y_train)
    
    # Evaluate model
    evaluation = model_dev.evaluate_model(X_test, y_test)
    print("Model Evaluation:", evaluation)

if __name__ == "__main__":
   def main():
    print("Main function is running...")
    # Simulated data generation (replace with your actual dataset)
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y = tf.keras.utils.to_categorical(np.random.randint(0, 5, 1000), 5)
    
    # Initialize model
    model_dev = AIModelDevelopment(input_shape=(10,), num_classes=5)
    
    # Prepare data
    X_train, X_test, y_train, y_test = model_dev.prepare_data(X, y)
    
    # Train model
    history = model_dev.train_model(X_train, y_train)
    
    # Evaluate model
    evaluation = model_dev.evaluate_model(X_test, y_test)
    print("Model Evaluation:", evaluation)
    
    # Save the trained model
    model_dev.model.save('E:/ai model/ai_model.h5')
    print("Model saved successfully!")
if __name__ == "__main__":
    main()
