# 🚗 Auto Insurance Fraud Detection

A comprehensive machine learning web application built with Streamlit for detecting fraudulent insurance claims using an Extra Trees Classifier model.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an intelligent fraud detection system for auto insurance claims. Using machine learning algorithms, it analyzes various claim parameters to predict whether a claim is fraudulent or legitimate, helping insurance companies reduce losses and improve claim processing efficiency.

## ✨ Features

- **🔍 Real-time Fraud Prediction**: Instant analysis of insurance claim data
- **📊 Interactive Dashboard**: User-friendly Streamlit interface
- **📋 Step-by-Step Input Mode**: Guided form filling experience
- **⚡ Quick Form Mode**: Bulk data entry for power users
- **🎲 Random Default Values**: Pre-filled realistic sample data
- **📈 Data Visualization**: Explore the training dataset
- **🔧 Input Validation**: Comprehensive form validation and error handling
- **🧹 Clear Form Function**: Reset all inputs with one click

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: scikit-learn (Extra Trees Classifier)
- **Data Processing**: Pandas, NumPy
- **Model Serialization**: Joblib
- **Deployment**: Streamlit Cloud

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/krrish-anand/Auto_Fraud_Detection.git
   cd Auto_Fraud_Detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   streamlit run app/app.py
   ```

6. **Open in browser**
   ```
   http://localhost:8501
   ```

## 💻 Usage

### Step-by-Step Mode
1. Select "📋 Step-by-Step" input mode
2. Fill out each field one by one
3. Use "➡️ Next" and "⬅️ Previous" buttons to navigate
4. Track progress with the completion bar
5. Click "🔍 Predict Fraud" when all fields are completed

### Quick Form Mode
1. Select "🏃 Quick Form" input mode
2. Fill out all fields simultaneously
3. Use "🗑️ Clear Values" to reset the form
4. Click "🔍 Predict Fraud" to get results

### Sample Prediction
The app comes with realistic default values that you can modify or use as-is to test the fraud detection model.

## 📁 Project Structure

```
Auto_Fraud_Detection/
├── app/
│   ├── app.py                      # Main Streamlit application
│   ├── utils/
│   │   ├── model_utils.py          # Model utility functions
│   │   └── __init__.py
│   └── __init__.py
├── data/
│   └── insurance_claims.csv        # Training dataset
├── models/
│   └── extra_trees_best_model.pkl  # Trained ML model
├── requirements.txt                # Python dependencies
├── .gitignore                     # Git ignore file
├── README.md                      # Project documentation
└── LICENSE                        # MIT License
```

## 🤖 Model Information

### Algorithm
- **Model Type**: Extra Trees Classifier (Extremely Randomized Trees)
- **Framework**: scikit-learn 1.6.1
- **Training Features**: 35 features including customer demographics, policy details, incident information, and vehicle characteristics

### Key Features Used
- Customer demographics (age, education, occupation)
- Policy information (state, CSL, deductible, premium)
- Incident details (type, severity, location, time)
- Vehicle information (make, model, year)
- Claim amounts (total, injury, property, vehicle)

### Model Performance
The Extra Trees model was selected for its:
- High accuracy in fraud detection
- Robustness against overfitting
- Fast prediction speed
- Good handling of mixed data types

## 📸 Screenshots

### Main Dashboard
![Main Dashboard](https://via.placeholder.com/800x400?text=Streamlit+Fraud+Detection+Dashboard)

### Step-by-Step Mode
![Step-by-Step Mode](https://via.placeholder.com/800x400?text=Step-by-Step+Input+Mode)

### Prediction Results
![Prediction Results](https://via.placeholder.com/800x400?text=Fraud+Prediction+Results)

## 🔮 Future Enhancements

- [ ] Model retraining pipeline
- [ ] A/B testing framework
- [ ] Advanced data visualization
- [ ] API endpoint for batch predictions
- [ ] Model explainability features (SHAP values)
- [ ] Historical prediction tracking
- [ ] Email notifications for fraud alerts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**Krrish Anand** - [@krrish-anand](https://github.com/krrish-anand)

Project Link: [https://github.com/krrish-anand/Auto_Fraud_Detection](https://github.com/krrish-anand/Auto_Fraud_Detection)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Insurance industry best practices for fraud detection
- Streamlit community for the amazing framework
- scikit-learn team for the machine learning tools
- Open source community for inspiration and support

---

⭐ If you found this project helpful, please give it a star on GitHub!