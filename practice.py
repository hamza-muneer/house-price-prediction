import pandas as pd
def main():
    data = pd.read_csv('train.csv')
    lot_frontage_mean = round(data['LotFrontage'].mean())
    data['LotFrontage'] = data['LotFrontage'].fillna(lot_frontage_mean)
    if data['MSSubClass']>120:
        data['MSSubClass'] = data['MSSubClass'].replace(data['MSSubClass'], 120)
        print("Values in 'MSSubClass' greater than 120 have been replaced with 120.")

    data.to_csv('train_cleaned.csv', index=False)
    print("Missing values in 'LotFrontage' have been filled with the mean and saved to 'train_cleaned.csv'.")

    
    
    
if __name__ == "__main__":  
    main()