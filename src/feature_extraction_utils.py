import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F


class FeatureExtractor:
    def __init__(self):
        self.model = None
        self.target_layer = None
        self.target_module_id = None
        self.features = None
        self.gradients = None
        self.input_tensor = None
        self.hook = None
        self.grad_hook = None

    def set_model(self, model):
        self.model = model
        self.model.eval()

    def set_target_layer(self, target_layer, target_module_id=None):
        self.model.eval()
        self.target_layer = target_layer
        self.target_module_id = target_module_id if target_module_id is not None else id(target_layer)
        self.features = None
        self.gradients = None

    def __save_features_hook(self, module, input, output):
        if id(module) == self.target_module_id:
            print(module)
            self.features = output.clone().detach()

    def __save_grads_hook(self, module, grad_input, grad_output):
        if id(module) == self.target_module_id:
            print(module)
            self.gradients = grad_output[0].clone().detach()

    def register_hooks(self):
        self.hook = self.target_layer.register_forward_hook(self.__save_features_hook)
        self.grad_hook = self.target_layer.register_full_backward_hook(self.__save_grads_hook)

    def set_input_tensor(self, input_tensor, label=None, see_picture=False):
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("Input tensor must be a torch.Tensor.")
        if input_tensor.dim() != 3:
            raise ValueError("Input tensor must be a 3D tensor (channels, height, width).")

        self.input_tensor = input_tensor.unsqueeze(0).requires_grad_()  # shape: [1, C, H, W]

        if see_picture:
            img_np = input_tensor.clone().permute(1, 2, 0).numpy()
            plt.imshow(img_np)
            plt.title(f"Label: {label}")
            plt.axis('off')
            plt.show()

    def extract_features(self):
        self.model.eval()
        if self.model is None or self.target_layer is None:
            raise ValueError("Model and target layer must be set before extracting features.")
        if self.input_tensor is None:
            raise ValueError("Input tensor must be set before extracting features.")

        device = next(self.model.parameters()).device
        self.input_tensor = self.input_tensor.to(device)
        self.input_tensor = self.input_tensor.detach().clone().requires_grad_()
        self.model = self.model.to(device)

        # Forward pass
        output = self.model(self.input_tensor)
        pred_class = output.argmax(dim=1).item()

        # Backward pass for Grad-CAM
        self.model.zero_grad()
        class_score = output[0, pred_class]
        class_score.backward()

        # Remove hooks
        if self.hook is not None:
            self.hook.remove()
        if self.grad_hook is not None:
            self.grad_hook.remove()

        return self.features.cpu(), self.gradients.cpu(), pred_class


def grad_cam(features, gradients, subset_features=None):
    """
    Generate Grad-CAM heatmap from features and gradients.
    :param features: Output features from the target layer.
    :param gradients: Gradients from the backward pass.
    :return: Heatmap as a numpy array.
    """

    # Grad-CAM weights (global average pooling over gradients)
    avg_gradients = gradients.mean(dim=(2, 3), keepdim=True)  # shape: [1, C, 1, 1]

    # Weighted sum of feature maps
    if subset_features is not (None):
        gradcam = torch.relu(avg_gradients[:, subset_features] * features[:, subset_features]).sum(dim=1,
                                                                                                   keepdim=True)  # shape: [1, 1, H, W]
        alpha = avg_gradients.squeeze()[subset_features]
    else:
        gradcam = torch.relu(avg_gradients[:, :] * features[:, :]).sum(dim=1, keepdim=True)  # shape: [1, 1, H, W]
        alpha = avg_gradients

    # Normalize
    gradcam -= gradcam.min()
    gradcam /= (gradcam.max() - gradcam.min() + 1e-8)  # Avoid division by zero

    return gradcam.detach().cpu().numpy(), alpha.squeeze().detach().numpy()  # Convert to numpy array for visualization


def show_cam_on_image(input_tensor, heatmap, alpha=0.5):
    # Convert input_tensor (CHW) to PIL image
    image = transforms.ToPILImage()(input_tensor)

    # We need to match the input image size
    heatmap = np.uint8(255 * heatmap.squeeze())  # Shape: (H, W)
    heatmap = Image.fromarray(heatmap).resize(image.size, resample=Image.BILINEAR)

    # Apply color map
    heatmap = np.array(heatmap) / 255.0
    colormap = plt.cm.jet(heatmap)[..., :3]  # RGB only

    # Overlay with input image
    image_np = np.array(image) / 255.0
    overlay = alpha * colormap + (1 - alpha) * image_np
    overlay = np.clip(overlay, 0, 1)

    # Plot
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()


def show_cam_as_intensity_mask(input_tensor, heatmap):
    # Convert input tensor (CHW) to PIL Image and then numpy RGB [0,1]
    image = transforms.ToPILImage()(input_tensor)
    image_np = np.array(image).astype(np.float32) / 255.0  # shape (H, W, 3)

    # Resize heatmap to image size and normalize [0,1]
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap.squeeze()))
    heatmap_resized = heatmap_resized.resize(image.size, resample=Image.BILINEAR)
    heatmap_np = np.array(heatmap_resized).astype(np.float32) / 255.0  # shape (H, W)

    # Expand heatmap to 3 channels for RGB masking
    heatmap_3ch = np.stack([heatmap_np] * 3, axis=-1)  # shape (H, W, 3)

    # Mask the image by the heatmap intensity (element-wise multiply)
    masked_image = image_np * heatmap_3ch

    # Show masked image
    plt.imshow(masked_image)
    plt.axis('off')
    plt.show()


def cosine_similarity(features):
    cov = torch.zeros((features.shape[0], features.shape[0]))
    for a in range(features.shape[0]):
        for b in range(features.shape[0]):
            cov[a, b] = F.cosine_similarity(torch.from_numpy(features[a]).flatten(),
                                            torch.from_numpy(features[b]).flatten(), dim=0)

    cov = cov.abs()  # Ensure covariance is non-negative
    cov.diagonal().fill_(0)  # Set diagonal to zero to ignore self-similarity

    return cov.detach().numpy()


def run_gradcam_hooks(model, target_layer, input_tensor, device="cpu", show_image=True):
    """
    Run a forward and backward pass on an input from val_dataset with the specified target label,
    while collecting the feature maps and gradients from the specified layer (via hooks).

    Returns:
        input_tensor: the selected input image tensor
        features: feature map from the target layer
        gradients: gradients of the output w.r.t. features
        pred_class: predicted class index
    """
    features = None
    target_module_id = id(target_layer)

    def save_features_hook(module, input, output):
        nonlocal features
        if id(module) == target_module_id:
            print(f"Captured from target layer: {module}")
            features = output.clone()  # Clone to avoid in-place modification

    gradients = None

    def save_grads_hook(module, grad_input, grad_output):
        nonlocal gradients
        if id(module) == target_module_id:
            gradients = grad_output[0]  # shape: [1, C, H, W]

    # Register hooks
    f_hook = target_layer.register_forward_hook(save_features_hook)
    b_hook = target_layer.register_full_backward_hook(save_grads_hook)

    # Select a random input with the given label
    np.random.seed(15)
    torch.manual_seed(42)

    input_tensor.requires_grad_()

    # Optionally show image
    if show_image:
        img_np = input_tensor.clone().squeeze().permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(img_np)
        plt.axis('off')
        plt.show()

    # Requires gradients for Grad-CAM
    input_tensor.requires_grad_()
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    # Forward pass
    output = model(input_tensor)  # shape: [1, 1000]

    pred_class = output.argmax(dim=1).item()

    print("Predicted class:", pred_class)

    # label_name = dataset.classes[pred_class]
    # print(f"Label index: {pred_class}, name: {label_name}")

    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    # Remove hooks
    f_hook.remove()
    b_hook.remove()

    return features, gradients, pred_class