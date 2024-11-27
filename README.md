# Youth Mental Health Prediction Project

## Overview

The **Youth Mental Health Prediction Project** aims to leverage machine learning models to predict various aspects of youth mental health based on provided features. The project utilizes advanced natural language processing techniques, including Masked Language Modeling (MLM) and sequence classification with BERT-based models, to generate accurate predictions. This documentation provides an overview of the project structure, installation instructions, usage guidelines, key components, lessons learned, and contribution guidelines.

## Directory Structure

```
.
├── .gitignore
├── .vscode/
│   └── [launch.json](.vscode/launch.json)
├── model_training/
│   ├── [feature_classification_input.json](model_training/feature_classification_input.json)
│   ├── notebooks/
│   │   ├── [Bert-base-uncased-finetuning.ipynb](model_training/notebooks/Bert-base-uncased-finetuning.ipynb)
│   │   ├── [EDA.ipynb](model_training/notebooks/EDA.ipynb)
│   │   └── ...
│   ├── private_data/
│   │   └── ...
│   └── scripts/
│       └── ...
├── [README.md](README.md)
├── [requirements.txt](requirements.txt)
├── [setup.py](setup.py)
└── youth-mental-health-runtime/
    ├── [.dockerignore](youth-mental-health-runtime/.dockerignore)
    ├── .vscode/
    │   └── ...
    ├── [CHANGELOG.md](youth-mental-health-runtime/CHANGELOG.md)
    ├── data/
    │   └── ...
    ├── [dev-requirements.txt](youth-mental-health-runtime/dev-requirements.txt)
    ├── example_submission/
    ├── [gitignore](youth-mental-health-runtime/gitignore)
    ├── images/
    ├── json/
    ├── [LICENSE](youth-mental-health-runtime/LICENSE)
    ├── [MAINTAINERS.md](youth-mental-health-runtime/MAINTAINERS.md)
    ├── [Makefile](youth-mental-health-runtime/Makefile)
    ├── pd/
    ├── [pyproject.toml](youth-mental-health-runtime/pyproject.toml)
    ├── [README.md](youth-mental-health-runtime/README.md)
    ├── runtime/
    ├── src/
    ├── submission/
    ├── submission_src/
    └── torch/
```

## Installation

1. **Clone the Repository**
   ```sh
   git clone https://github.com/kaungkhantko/YouthMentalHealthNarratives-AutomatedAbstraction-SavingPrivateNYH.git
   cd youth-mental-health-prediction
   ```

2. **Set Up Virtual Environment**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```sh
   pip install -r youth-mental-health-runtime/dev-requirements.txt
   ```

4. **Docker Setup (Optional)**
   If you prefer using Docker, ensure Docker is installed and build the Docker image:
   ```sh
   docker build -t youth-mental-health-runtime youth-mental-health-runtime/
   ```

## Usage

### Running Inference

To generate predictions using the MLM model:
```sh
python youth-mental-health-runtime/submission_src/main_MLM.py
```

For sequence classification with BERT:
```sh
python youth-mental-health-runtime/submission_src/main_BERTForSequenceClassification.py
```

### Model Training

Navigate to the 

model_training

 directory and run the desired training script. For example, to fine-tune the BERT model:
```sh
jupyter notebook model_training/notebooks/Bert-base-uncased-finetuning.ipynb
```

### Evaluation

Use the provided `scoring.py` script to evaluate your predictions:
```sh
python youth-mental-health-runtime/src/scoring.py submission/submission.csv data/train_labels.csv
```

## Key Components

### 

model_training



- **`feature_classification_input.json`**: Defines the input features for classification tasks.
- **notebooks/**
  - **

Bert-base-uncased-finetuning.ipynb

**: Notebook for fine-tuning the BERT model.
  - **

EDA.ipynb

**: Exploratory Data Analysis notebook.
  - ...
- **private_data/**: Contains sensitive or proprietary data not included in version control.
- **scripts/**: Python scripts for automating training processes.

### 

youth-mental-health-runtime



- **`.dockerignore`**: Specifies files to ignore when building Docker images.
- **.vscode/**: VSCode-specific settings and configurations.
- **

CHANGELOG.md

**: Documentation of all significant changes to the project.
- **data/**: Runtime-specific data files.
- **

dev-requirements.txt

**: Development dependencies.
- **example_submission/**: Example code for submissions.
- **`gitignore`**: Files and directories to be ignored by Git in the runtime environment.
- **images/**: Images used in documentation and README files.
- **json/**: JSON configuration files.
- **`LICENSE`**: Licensing information for the project.
- **

MAINTAINERS.md

**: Maintainer notes and guidelines.
- **`Makefile`**: Automation of common tasks using `make` commands.
- **pd/**: Placeholder for specific project directories or files.
- **

pyproject.toml

**: Project configuration for build tools.
- **

README.md

**: Documentation specific to the runtime environment.
- **runtime/**: Contains runtime-specific configurations and scripts.
- **src/**: Source code for the runtime environment.
- **submission/**: Directory for submission files.
- **submission_src/**: Source code for actual submissions.
- **torch/**: PyTorch-related files and configurations.

## Lessons Learned

Throughout the development of the Youth Mental Health Prediction Project, several key lessons were learned that have significantly influenced the project's direction and implementation strategies. This section outlines the most notable insights and takeaways.

### 1. **Importance of Data Integrity and Preprocessing**
- **Insight:** Ensuring data quality is paramount for model performance.
- **Action Taken:** Implemented rigorous data cleaning and preprocessing pipelines to maintain data integrity.

### 2. **Robust Exception Handling Enhances Reliability**
- **Insight:** Proper error handling prevents unexpected crashes and aids in debugging.
- **Action Taken:** Added comprehensive exception handling across all modules to enhance system reliability.

### 3. **Effective Logging is Crucial for Debugging and Monitoring**
- **Insight:** Without adequate logging, tracking the flow of data and identifying points of failure becomes challenging.
- **Action Taken:** Established a detailed logging mechanism that captures debug information, warnings, and errors.

### 4. **Model Output Standardization Improves Mapping Accuracy**
- **Insight:** Variability in model outputs can hinder accurate mapping to predefined labels.
- **Action Taken:** Implemented output cleaning steps to standardize model predictions before mapping them to label values.

### 5. **Separation of Concerns Enhances Code Maintainability**
- **Insight:** Mixing different functionalities within the same module makes the codebase convoluted.
- **Action Taken:** Structured the code to separate data loading, prediction generation, and result aggregation into distinct functions.

### 6. **Ensuring Alignment Between Data Sources Prevents Inconsistencies**
- **Insight:** Misalignment between different data sources can introduce inconsistencies and errors.
- **Action Taken:** Ensured that features are exclusively sourced from `test_features.csv` and initialized predictions based on this dataset's UIDs.

### 7. **Performance Optimization is Key for Scalability**
- **Insight:** Optimizing model performance is essential for handling larger datasets and reducing inference time.
- **Action Taken:** Applied various optimization techniques to improve model efficiency and scalability.

### 8. **Comprehensive Understanding of AI/ML Concepts**
- **Insight:** A deep understanding of AI/ML concepts is essential for building effective models.
- **Action Taken:** Continuously studied and applied advanced AI/ML techniques to enhance model performance.

### 9. **LLM Prompting Techniques Enhance Model Performance**
- **Insight:** Effective prompting techniques significantly influence the performance of Large Language Models (LLMs).
- **Action Taken:** Experimented with various prompting strategies to determine the most effective approaches for different prediction tasks.

### 10. **Mastery of Hugging Face Libraries Facilitates Model Deployment**
- **Insight:** Proficiency in Hugging Face libraries streamlines model loading, tokenization, and integration processes.
- **Action Taken:** Utilized various Hugging Face libraries to enhance model deployment workflows.

### 11. **Understanding Containerization with Docker**
- **Insight:** Containerization ensures consistent environments across development, testing, and deployment.
- **Action Taken:** Mastered the use of Docker for building, managing, and deploying containerized applications.

### 12. **Grasping Dependency Management**
- **Insight:** Effective dependency management is crucial for maintaining project stability and avoiding conflicts.
- **Action Taken:** Managed project dependencies meticulously using 

requirements.txt

 and virtual environments.

### 13. **Deep Dive into Model Architectures**
- **Insight:** Exploring different model architectures provides a broader perspective on solving various NLP tasks.
- **Action Taken:** Implemented and tested different model architectures to determine their suitability for specific prediction tasks.

### 14. **Parameter-Efficient Fine Tuning (PEFT) for Resource Optimization**
- **Insight:** PEFT techniques allow for effective model fine-tuning with fewer parameters.
- **Action Taken:** Applied PEFT methods to fine-tune models, achieving optimal performance without overburdening computational resources.

### 15. **Continuous Integration and Continuous Deployment (CI/CD) Practices**
- **Insight:** Implementing CI/CD pipelines automates testing, integration, and deployment processes.
- **Action Taken:** Established CI/CD workflows to automate testing and deployment, ensuring smooth integration and reliable deployments.

### 16. **Effective Use of Hugging Face in Development Environments**
- **Insight:** Managing access to Hugging Face repositories is essential for seamless model integration.
- **Action Taken:** Managed authentication and repository access effectively within development environments.

### 17. **Question-Answering and Fill-Mask Model Implementations**
- **Insight:** Diverse NLP tasks benefit from specialized model architectures.
- **Action Taken:** Implemented question-answering and fill-mask models to enhance the project's prediction capabilities.

### 18. **Proper Debugging and Logging Techniques for AI Deployment**
- **Insight:** Effective debugging and logging are essential for maintaining and monitoring deployed models.
- **Action Taken:** Implemented advanced debugging and logging techniques to ensure smooth deployment and monitoring of AI models.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a New Branch**
   ```sh
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**
   ```sh
   git commit -m "Add your feature"
   ```

4. **Push to the Branch**
   ```sh
   git push origin feature/YourFeature
   ```

5. **Create a Pull Request**

For detailed guidelines, refer to the CONTRIBUTING.md file.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Contact

For questions or support, please contact [k.khantko@outlook.com](mailto:k.khantko@outlook.com).

## Acknowledgements

- [Transformers by Hugging Face](https://github.com/huggingface/transformers)
- [pandas](https://pandas.pydata.org/)
- [PyTorch](https://pytorch.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [Pixi](https://pixi.sh/)
- [LangChain](https://github.com/langchain-ai/langchain)

## References

- [Youth Mental Health Data](https://www.drivendata.org/competitions/295/cdc-automated-abstraction/data/)
- [Machine Learning Best Practices](https://www.drivendata.org/competitions/295/cdc-automated-abstraction/page/923/#best-practices)

## Getting Started

To get started with the Youth Mental Health Prediction Project:

1. **Ensure all dependencies are installed** by following the Installation section.
2. **Set up the data directory** as outlined in the Quickstart guide within the 

youth-mental-health-runtime

 directory.
3. **Run inference or train models** following the Usage guidelines.

## Support

If you encounter any issues or have suggestions for improvements, please open an issue in the repository or contact the maintainer.

## Conclusion

The Youth Mental Health Prediction Project integrates various machine learning models to provide insightful predictions that can aid in understanding and addressing youth mental health challenges. By following this documentation, contributors and users can effectively navigate and utilize the project's capabilities.
