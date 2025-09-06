# Wine Quality Prediction with KNN 🍷

---

## Description

This project predicts **wine quality** using the **k-nearest neighbor (KNN)** algorithm and recommends wines based on user inputs.
It uses the **Red Wine Quality dataset** from Kaggle and includes visualizations for data analysis and model performance.

* Dataset: [Red Wine Quality (Kaggle)](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download)
* Author: **Begüm Şara Ünal**

---

## Features

* Explore the dataset with **pandas**
* Visualize distributions and relationships with **Seaborn** and **Matplotlib**
* Preprocess data with **StandardScaler**
* Train a **KNN classifier** and evaluate accuracy
* Compare model performance across different values of **k**
* Support multiple distance metrics:

  * Cosine Similarity
  * Minkowski Distance
  * Euclidean Distance
  * Manhattan Distance
* Recommend the closest wine to user input

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/wine-quality-knn.git
cd wine-quality-knn
```

2. Install required packages:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download) and place `winequality-red.csv` in the project folder.

---

## Usage

1. Run the script:

```bash
python wineQualityWithKNN.py
```

2. The script will:

   * Display dataset info
   * Train KNN model and print accuracy
   * Plot accuracy vs. k
   * Accept user input to recommend a wine

3. Example input:

```
Fixed Acidity: 7.4
Volatile Acidity: 0.7
Citric Acid: 0.0
Residual Sugar: 1.9
Chlorides: 0.076
Free Sulfur Dioxide: 11.0
Total Sulfur Dioxide: 34.0
Density: 0.9978
PH Value: 3.51
Sulphates: 0.56
Alcohol: 9.4
```

---

## Visualizations 📊

### Accuracy vs K Values

![Accuracy vs K](images/accuracy_vs_k.png)

### Distance Metrics vs Index

* **Cosine Similarity**
  ![Cosine Similarity](images/cosine_similarity.png)

* **Minkowski Distance**
  ![Minkowski Distance](images/minkowski_distance.png)

* **Euclidean Distance**
  ![Euclidean Distance](images/euclidean_distance.png)

* **Manhattan Distance**
  ![Manhattan Distance](images/manhattan_distance.png)

> 💡 Tip: Save your plots as PNGs in `images/` folder using `plt.savefig("images/plot_name.png")`.

---

## Distance Metrics

| Metric    | Description                                    |
| --------- | ---------------------------------------------- |
| Cosine    | Measures similarity between vectors            |
| Minkowski | Generalized distance metric with parameter `p` |
| Euclidean | Minkowski distance with `p=2`                  |
| Manhattan | Minkowski distance with `p=1`                  |

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Contact

* Author: **Begüm Şara Ünal**
* Email: *\[[begumsaraunal@gmail.com](mailto:[begumsaraunal@gmail.com)]*
