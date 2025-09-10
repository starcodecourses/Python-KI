import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def plot_sample_images(X, y, num_samples=5):
    plt.figure(figsize=(10, 5))
    
    for i in range(num_samples):
        # Select a random index
        index = np.random.randint(0, len(X))

        # Plot image
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[index].reshape(28, 28), cmap='gray')
        plt.axis('off')

        # Show the label
        plt.title(f'Label: {y[index]}')

    plt.tight_layout()
    plt.show()

def beispielbilder_anzeigen(X, y, num_samples=5):
    plt.figure(figsize=(10, 5))
    
    for i in range(num_samples):
        index = np.random.randint(0, len(X))
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[index].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(f'Label: {y[index]}')

    plt.tight_layout()
    plt.show()


# Train-test split visualization
def plot_train_test_split_pie(X_train, X_test):
    train_size = len(X_train)
    test_size = len(X_test)

    # Define the labels and sizes
    labels = ['Train Set', 'Test Set']
    sizes = [train_size, test_size]
    colors = ['#66c2a5', '#8da0cb']  # Kept two of the original colors

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct=lambda p: f'{p:.1f}%\n({int(p / 100 * sum(sizes))})', 
        startangle=140, 
        colors=colors, 
        wedgeprops=dict(width=0.4)
    )

    # Customize text appearance
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(12)

    # Add a title
    plt.title('Train-Test Split Proportions', fontsize=16, fontweight='bold')
    plt.setp(autotexts, size=10, weight="bold")

    # Create a legend
    legend_elements = [Patch(facecolor=colors[0], label=f'Train Set ({train_size} samples)'),
                       Patch(facecolor=colors[1], label=f'Test Set ({test_size} samples)')]
    ax.legend(handles=legend_elements, loc='best', fontsize=12)

    # Display the plot
    plt.show()

# Plotting accuracy and loss
def plot_training_history(history):
    # Get the values from the training history
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)

    # Plot training & validation accuracy
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def trainingsverlauf_visualisieren(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Trainingsgenauigkeit')
    plt.title('Trainings- und Validierungsgenauigkeit')
    plt.xlabel('Epochen')
    plt.ylabel('Genauigkeit')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Trainingsverlust')
    plt.title('Trainings- und Validierungsverlust')
    plt.xlabel('Epochen')
    plt.ylabel('Verlust')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_predictions(X_test, y_test, predicted_labels):
    # Plot some of the test images with their predicted and actual labels
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'Predicted: {predicted_labels[i]}\nActual: {y_test[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def vorhersagen_visualisieren(X_test, y_test, predicted_labels):
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'Vorhergesagt: {predicted_labels[i]}\nTatsächlich: {y_test[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def visualize_false_predictions(X_test, y_test, predicted_labels, incorrect_indices, num_errors_to_display=8):
    # Create a figure with a 2x4 grid layout
    fig, axes = plt.subplots(2, 4, figsize=(10, 8))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i in range(min(num_errors_to_display, len(incorrect_indices))):
        idx = incorrect_indices[i]
        axes[i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Predicted: {predicted_labels[idx]}\nActual: {y_test[idx]}')
        axes[i].axis('off')

    # Turn off any remaining unused axes
    for j in range(num_errors_to_display, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def falsche_vorhersagen_visualisieren(X_test, y_test, predicted_labels, incorrect_indices, num_errors_to_display=8):
    fig, axes = plt.subplots(2, 4, figsize=(10, 8))
    axes = axes.flatten()

    for i in range(min(num_errors_to_display, len(incorrect_indices))):
        idx = incorrect_indices[i]
        axes[i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Vorhergesagt: {predicted_labels[idx]}\nTatsächlich: {y_test[idx]}')
        axes[i].axis('off')

    for j in range(num_errors_to_display, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()