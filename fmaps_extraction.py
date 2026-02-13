import numpy as np
import torch
import os
from tqdm import tqdm
from src.feature_extraction_utils import FeatureExtractor, grad_cam


##### Load test dataset and the model
cwd = os.getcwd()
test_dataset=torch.load(os.path.join(cwd, "dataset", "test_dataset.pt"), weights_only=False)
n_feature_last_block = 64
model = torch.load(os.path.join(cwd, "trained_model", f"model_{n_feature_last_block}_features_trained_20_epochs"), map_location="cpu", weights_only=False)
model.eval()


## we store only the correectly classified images
ref_labels = np.arange(10)
class_inputs = []

for ref_label in ref_labels:
    inputs = []
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        if label == ref_label:
            inputs.append(img)

    class_inputs.append(inputs)

## we select a reasonable number of images for each class (N_tot = 20 * 10 = 200)
n_samples = 20
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
                f"model{n_feature_last_block}_fmaps"
            )

            os.makedirs(save_dir, exist_ok=True)

            file_path = os.path.join(
                save_dir,
                f"class_{r}_idx_{idx}.npz"
            )

            np.savez(
                file_path,
                class_id=r,
                dataset_idx=idx,
                pred=pred,
                pos_indices=sorted_indices,
                alpha=alpha
            )