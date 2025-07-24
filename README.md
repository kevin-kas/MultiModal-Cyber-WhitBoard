# Multi-Modal Cyber Whiteboard: README

## Project Overview
Cyber Whiteboard is a deep learning-based multimodal mathematical interaction system designed to address the dual challenge of recognizing and generating handwritten mathematical expressions. The system enables bidirectional conversion between handwritten math formulas and digital representations such as LaTeX, referred to as Cyber Whiteboard . It supports both simple expression calculation and complex expression processing, with a user-friendly front-end interface for students and other users needing to digitize mathematical content .

## Core Features
- **Handwritten Expression Recognition**: Identifies mathematical symbols from handwritten input, supporting operations like addition, subtraction, multiplication, brackets, fractions, square roots, and exponents for simple expressions, and converts complex expressions with variables into LaTeX format .
- **Mathematical Calculation**: Performs computations on recognized simple arithmetic expressions to return results.
- **Handwritten Symbol Generation**: Generates readable handwritten-style symbols from digital inputs using VAE and diffusion models .
- **Interactive Interface**: A web-based front-end allowing users to input handwritten formulas, trigger recognition/calculation, and view generated handwritten symbols .

## Technical Architecture
### 1. Dataset & Preprocessing
- The dataset consists of single-channel (black-and-white) mathematical symbol images from Kaggle .
- Preprocessing includes balanced sampling for class imbalance, pixel normalization, PCA for dimensionality reduction, and an 80:20 train-test split .

### 2. Recognition Models
The system implements six classification models for recognition:
- **Baseline Models**: Gaussian Naïve Bayes (GNB), Extremely Randomized Trees (ERT), Multi-layer Perceptron (MLP), and Support Vector Machine (SVM) .
- **Deep Learning Models**: CNN for symbol recognition and structure analysis, and CAN (Counting-Aware Network) for direct LaTeX generation .

| Model         | Test Set Accuracy | Key Advantage                                  |
|---------------|-------------------|------------------------------------------------|
| GNB           | 69%               | Probability-based classification               |
| PCA+SVM       | 91%               | Handles nonlinear boundaries with RBF kernel   |
| CNN           | 99%               | Captures 2D spatial features effectively       |
| CAN           | 54% (standard)    | Optimizes complex structure recognition with counting auxiliary task  |

### 3. Generation Models
Two generative models are implemented:
- **VAE (Variational Auto-Encoder)**: Encodes images and labels into latent space, then decodes to reconstruct handwritten symbols with class control .
- **Diffusion Model**: Uses forward noising and reverse denoising processes, with U-Net, attention, and residual blocks to generate high-quality handwritten symbols .  
The diffusion model outperforms VAE, with a lower FID score (≈70 vs. ≈126) between real and generated images .

## Usage Guide
### Environment Requirements
- **Front-end**: Modern browser supporting HTML5, CSS3, and JavaScript .
- **Back-end**: Python (dependencies vary by model implementation).

### Interface Operations
1. Access the start page and review user guidance .
2. Use the "whiteboard" area to input handwritten formulas .
3. Use Button:
   - "Get value": Obtain calculation results for simple expressions .
   - "Get line": Convert complex formulas to LaTeX format .
   - "VAE generation"/"DIF generation": Generate handwritten symbols from input numbers .

## Future Work
- **Recognition Enhancement**: Integrate ReNet, ResNet, and LeNet to improve spatial dependency capture and deep feature extraction .
- **Generation Enhancement**: Develop a multi-stage pipeline with graph-based alignment for end-to-end complex expression generation .
- **Multimodal Expansion**: Add speech recognition with cross-modal Transformer for joint speech-handwriting understanding .

## Acknowledgments
The dataset is sourced from Kaggle. The project builds on research in Handwritten Mathematical Expression Recognition (HMER) and generative models .
