# Gas Morphology Classifier for Binary Evolution

This project aims to build a machine learning pipeline that classifies gas morphologies around binary star systems using synthetic images inspired by JWST observations. These morphologies, such as spirals, shells, and bipolar outflows, contain rich signatures of past and ongoing binary interactions (e.g., Roche lobe overflow, common envelope evolution, or jet activity).

This is Phase 1 of a multi-phase effort to connect resolved gas structures to the evolutionary states of binary stars and their likelihood of becoming gravitational wave (GW) progenitors. In this phase, the goal is to create a robust classifier trained on simulation-informed mock observations of dusty binaries in diverse environments.

1. **Generate synthetic data** with realistic morphologies and metadata.
2. **Preprocess and augment** images for robust model training.
3. **Train a deep CNN** to classify gas morphologies, using a reproducible config.
4. **Evaluate the model** with rigorous metrics and visualize learned representations.
5. **All code is modular, documented, and ready for extension** (e.g., real JWST data, new morphologies, interpretability tools).

## **JWST NIRCam Gas Morphology Classification Pipeline: Summary**

### **1. Synthetic Data Generation**
- **Morphology Classes:** spiral, shell, outflow, irregular, no_gas
- **Image Format:** 4-channel (F200W, F277W, F356W, F444W), 128×128 pixels, `.npy` files
- **Metadata:** Each image is paired with RA, Dec, environment, distance, and class label in a CSV
- **Organization:** `data/raw/<class_name>/<img_id>.npy`

### **2. Data Preprocessing & Augmentation**
- **Normalization:** Each image channel is normalized to [0, 1]
- **Augmentations:** Random rotation, flip, and Gaussian noise for robust training
- **PyTorch Dataset:** Loads images and metadata, applies augmentations, returns (tensor, label)
- **DataLoader:** Efficient batching, shuffling, and multiprocessing

### **3. Model Training**
- **Architecture:** Custom 4-channel CNN (SimpleMorphCNN) with batch norm and dropout
- **Configurable:** All hyperparameters and paths in `src/phase1_config.yaml`
- **Training Loop:** Cross-entropy loss, Adam optimizer, learning rate scheduler, GPU support
- **Checkpointing:** Best model saved as `best_model.pt` by validation accuracy

### **4. Evaluation & Visualization**
- **Metrics:** Accuracy, F1 score, per-class precision/recall, confusion matrix
- **Latent Space:** t-SNE projection of final-layer features for visualizing class separability
- **Publication-Quality Figures:** High-res PNG and SVG for confusion matrix and t-SNE
- **Script:** `src/evaluate.py` (fully commented, modular, and reproducible)

### **5. Documentation & Reproducibility**
- **Code Comments:** Every function and block is documented for clarity
- **Config File:** All settings in one YAML for easy experiment tracking
- **Results Directory:** All figures and outputs saved in `results/`

**Figure 1. Confusion Matrix for Gas Morphology Classification.**  
The confusion matrix shows the performance of the trained CNN in distinguishing between five synthetic gas morphologies (spiral, shell, outflow, irregular, no_gas) in JWST NIRCam-like images. Diagonal elements indicate correct classifications, while off-diagonal elements highlight common misclassifications. The model achieves high accuracy for spiral and shell morphologies, with most confusion occurring between outflow and irregular classes, likely due to their similar spatial features.

**Figure 2. t-SNE Projection of Latent Features.**  
t-SNE visualization of the final-layer embeddings reveals clear clustering by morphology class, indicating that the CNN has learned discriminative features for each gas morphology. Overlap between outflow and irregular clusters is consistent with confusion matrix results, suggesting these classes are more challenging to separate.
