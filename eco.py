from sklearn.tree import DecisionTreeRegressor

class HousePricePredictor:
    def __init__(self, features, prices):
        self.model = DecisionTreeRegressor()
        self.model.fit(features, prices)

    def predict(self, feature_vector):
        return self.model.predict([feature_vector])[0]

def main():
    # Example data
    features = [[1000, 3, 2], [2000, 4, 3], [1500, 3, 2],
                [1800, 3, 2], [2200, 4, 3], [1300, 2, 2],
                [1700, 3, 2], [1900, 4, 3], [1600, 3, 2],
                [2100, 4, 3], [2500, 5, 3], [3000, 6, 4],
                [2700, 5, 3], [2800, 4, 3], [3200, 6, 4],
                [2300, 4, 3], [2400, 5, 3], [2600, 4, 3],
                [2900, 5, 3], [3100, 6, 4], [3300, 6, 4],
                [3500, 7, 5], [3700, 8, 5], [3400, 7, 5],
                [3800, 8, 5], [3600, 7, 5], [3900, 8, 5],
                [4000, 9, 6], [4200, 10, 6], [4100, 9, 6],
                [4300, 10, 6], [4400, 11, 7], [4500, 12, 7],
                [4700, 11, 7], [4600, 12, 7], [4800, 13, 8],
                [4900, 14, 8], [5000, 13, 8], ]
    prices = [50000, 80000, 60000, 65000, 85000, 55000,
              70000, 75000, 62000, 83000, 90000, 120000,
              95000, 88000, 125000, 92000, 98000, 87000,
              105000, 130000, 140000, 150000, 145000, 160000,
              155000, 170000, 180000, 175000, 190000, 200000,
              210000, 205000, 220000, 230000, 225000, 240000,
              235000, 250000]  # Removed one extra label

    # Initialize house price predictor
    predictor = HousePricePredictor(features, prices)

    # Get user input for a new house
    print("Please enter the features of the house:")
    square_footage = float(input("Enter the square footage of the house: "))
    num_bedrooms = int(input("Enter the number of bedrooms: "))
    num_bathrooms = int(input("Enter the number of bathrooms: "))
    feature_vector = [square_footage, num_bedrooms, num_bathrooms]

    # Predict house price
    predicted_price = predictor.predict(feature_vector)
    print(f"Predicted house price: ₹{predicted_price:.2f}")

if __name__ == "__main__":
    main()
