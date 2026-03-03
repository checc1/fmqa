import numpy as np
import torch
import os
from tqdm import tqdm
from src.feature_extraction_utils import FeatureExtractor, grad_cam
import sys
import torch.nn as nn
from torchvision import models


device = torch.device("cpu")


class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes, reduced_channels):
        super(ModifiedResNet18, self).__init__()
        original_resnet = models.resnet18(weights="DEFAULT")

        # Keep all layers up to layer3 (optional for now)
        self.features = nn.Sequential(
            original_resnet.conv1,
            original_resnet.bn1,
            original_resnet.relu,
            original_resnet.maxpool,
            original_resnet.layer1,
            nn.ReLU(inplace=False),
            original_resnet.layer2,
            nn.ReLU(inplace=False),
            original_resnet.layer3,
            nn.ReLU(inplace=False),
            #original_resnet.layer4,
            #nn.ReLU(inplace=False),
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=reduced_channels),
            nn.ReLU(inplace=False),
        )
        #self.reduce =nn.Sequential(nn.Conv2d(256, reduced_channels, kernel_size=3,padding=1),nn.BatchNorm2d(reduced_channels),nn.ReLU(inplace=False))

        # Classifier

        self.classifier = nn.Linear(reduced_channels*14**2, num_classes) # we have a better resolution of 14x14 after the last conv layer

    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


##### Load test dataset and the model
cwd = os.getcwd()
test_dataset = torch.load(os.path.join(cwd, "dataset", "test_dataset.pth"), weights_only=False)
#fmaps = int(sys.argv[1]); epochs = int(sys.argv[2])
fmaps = 128; epochs = 16

model = ModifiedResNet18(num_classes=10, reduced_channels=fmaps)
state_dict = torch.load(
    os.path.join(cwd, "checkpoints", f"checkpoint_ep{epochs}_fullResnetMod{fmaps}.pth"),
    map_location=device
)
model.load_state_dict(state_dict['model_state_dict'])
model.to(device)

#model = torch.load(os.path.join(cwd, "checkpoints", f"checkpoint_ep{epochs}_fullResnetMod{fmaps}.pth"), map_location="cpu", weights_only=False)
model.eval()


## we store only the correectly classified images
ref_labels = np.arange(10)
class_inputs = []
correct_classified = {}

for ref_label in ref_labels:
    inputs = []
    count = 0
    correct_classified[str(ref_label)] = count
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        if label == ref_label:
            inputs.append(img)
            count += 1
            correct_classified[str(ref_label)] = count

    class_inputs.append(inputs)


print(correct_classified)
list_of_correct_classified = [correct_classified[str(ref_label)] for ref_label in ref_labels]
#print(list_of_correct_classified)
min_n = min(list_of_correct_classified)
print(min_n)

## we select a reasonable number of images for each class (N_tot = 20 * 10 = 200)
n_samples = min_n
features_selected_per_class = []
accuracy_per_class = []
optimal_solutions_per_class = []
for r, inputs in enumerate(class_inputs):

    features_selected = []
    overall_value = len(inputs[:n_samples])
    tbar = tqdm(enumerate(inputs[:n_samples]))

    accuracy_count = 0
    opt_solution = []
    for idx, img_tensor in tbar:

        input_tensor = img_tensor.clone()
        extractor = FeatureExtractor()
        extractor.set_model(model)
        extractor.set_target_layer(model.reduce[-1])  # Target layer from model
        #print("Output", model.reduce[-1])
        extractor.set_input_tensor(img_tensor, see_picture=False)

        extractor.register_hooks()

        features, gradients, pred = extractor.extract_features()

        if pred == ref_labels[r]:
            accuracy_count += 1

            # we consider only nonzero features because are the only ones that contribute to the gradcam (positive contribution to gradient)
            # Compute per-channel spatial mean on the first sample
            channel_means = features[0].mean(dim=(1, 2))  # Shape: [C]

            # Get mask of non-zero mean channels
            # nonzero_mask = channel_means != 0  # Shape: [C]

            # # Apply mask to features and gradients
            # features = features[:, nonzero_mask, :, :]
            # gradients = gradients[:, nonzero_mask, :, :]

            zero_mask = channel_means == 0
            gradients[:, zero_mask, :, :] = -10 ** -10

            # print(features.shape)

            # print(features.shape,gradients.shape)
            gradcam, alpha = grad_cam(features=features.cpu(), gradients=gradients.cpu(), subset_features=None)
            # print(gradcam.shape)
            # print(gradcam[0,0].shape,input_tensor.shape)
            # show_cam_on_image(input_tensor=input_tensor,heatmap=gradcam,alpha=0.7)
            # show_cam_as_intensity_mask(input_tensor=input_tensor,heatmap=gradcam[0,0])
            # print(alpha.shape)
            # plt.bar(np.arange(alpha.shape[0]),alpha)
            # plt.show()

            sorted_indices = np.argsort(-alpha)
            alpha = alpha[sorted_indices]
            save_dir = os.path.join(
                cwd,
                "fmaps",
                f"model{fmaps}_fmaps"
            )

            os.makedirs(save_dir, exist_ok=True)

            file_path = os.path.join(
                save_dir,
                f"class_{r}_idx_{idx}.npz"
            )

            np.savez(
                file_path,
                image=input_tensor.cpu().numpy(),
                class_id=r,
                dataset_idx=idx,
                pred=pred,
                pos_indices=sorted_indices,
                alpha=alpha,
                feature_maps = features.cpu().numpy(),
                gradients = gradients.cpu().numpy(),
                grad_cams = gradcam,
            )