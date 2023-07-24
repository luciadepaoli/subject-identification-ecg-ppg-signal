# Subjects identification using ECG and PPG signals recorded from user-friendly devices
## Project for the Human Data Analytics course at University of Padua.

**Authors**: Lucia Depaoli, Simone Mistrali

**Project Summary**: Using a chest strap band and a pulse-oximeter, we record ECG and PPG signals from $4$ different subjects. After a filtering operation, we perform a subject identification using tools such as CNN, k-NN, SVM and deep learning techniques.

**Abstract**: Electrocardiogram (ECG) and photoplenthysmogram (PPG) signals have caught researchers attention recently, for a various amount of reasons. A part from the medical one, these signals can be used in biometric authentication and biometric identification in order to achieve high level of security, allowing the users to stop relying on traditional procedure such as passwords or PINs, which can be easily broken down by cyberattacks. In this work, starting from two user-friendly devices that can records these signals, we construct two datasets. The first one is made up by data taken from 4 different subjects, while the second one data from 2 subjects, both at rest and in motion. Using different algorithm and CNN architectures, we managed to correctly classify the subjects and the motion, reaching an accuracy on the test set of about 99% for the ECG signal and 90% for the PPG signal.

For more information, see the [paper](https://github.com/luciadepaoli/subject-identification-ecg-ppg-signal/blob/main/report_human_data_depaoli_mistrali.pdf).

### File description
- `dataset_creation.py`: creation of ECG and PPG dataset directly from the file produced by the devices we have used.
- `at_rest_KNN.ipynb` and `in_motion_KNN.ipynb`: classification using K-NN.
- `at_rest_SVM.ipynb` and `in_motion_SVM.ipynb`: classification using SVM technique.
- `neural_network_ecg.ipynb`: classification using Keras neural networks.
- `t-SNE.ipynb`: data visualization using t-SNE.
