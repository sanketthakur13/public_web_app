import numpy as np
import pickle
import streamlit as st


#####################code for model############################
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

loaded_model = pickle.load(open('chicago_crime.sav','rb'))

# Dictionary to map string labels to encoded numbers for each column
label_mappings = {
    'LocationDescription': {'ABANDONED BUILDING': 71, 'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA': 61, 'AIRPORT BUILDING NON-TERMINAL - SECURE AREA': 39, 'AIRPORT EXTERIOR - NON-SECURE AREA': 56, 'AIRPORT EXTERIOR - SECURE AREA': 75, 'AIRPORT PARKING LOT': 29, 'AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA': 60, 'AIRPORT VENDING ESTABLISHMENT': 52, 'AIRPORT/AIRCRAFT': 9, 'ALLEY': 70, 'ANIMAL HOSPITAL': 74, 'APARTMENT': 58, 'APPLIANCE STORE': 67, 'ATHLETIC CLUB': 34, 'BANK': 57, 'BAR OR TAVERN': 68, 'BARBERSHOP': 11, 'BOWLING ALLEY': 76, 'BRIDGE': 7, 'CAR WASH': 15, 'CHA APARTMENT': 55, 'CHA PARKING LOT/GROUNDS': 42, 'CHURCH/SYNAGOGUE/PLACE OF WORSHIP': 24, 'CLEANING STORE': 69, 'COLLEGE/UNIVERSITY GROUNDS': 19, 'COLLEGE/UNIVERSITY RESIDENCE HALL': 36, 'COMMERCIAL / BUSINESS OFFICE': 62, 'CONSTRUCTION SITE': 37, 'CONVENIENCE STORE': 21, 'CTA GARAGE / OTHER PROPERTY': 3, 'CTA TRAIN': 30, 'CURRENCY EXCHANGE': 41, 'DAY CARE CENTER': 5, 'DEPARTMENT STORE': 49, 'DRIVEWAY - RESIDENTIAL': 72, 'DRUG STORE': 40, 'FACTORY/MANUFACTURING BUILDING': 51, 'FIRE STATION': 8, 'FOREST PRESERVE': 44, 'GAS STATION': 22, 'GOVERNMENT BUILDING/PROPERTY': 27, 'GROCERY FOOD STORE': 26, 'HIGHWAY/EXPRESSWAY': 66, 'HOSPITAL BUILDING/GROUNDS': 33, 'HOTEL/MOTEL': 77, 'JAIL / LOCK-UP FACILITY': 43, 'LAKEFRONT/WATERFRONT/RIVERBANK': 59, 'LIBRARY': 4, 'MEDICAL/DENTAL OFFICE': 73, 'MOVIE HOUSE/THEATER': 20, 'NEWSSTAND': 65, 'NURSING HOME/RETIREMENT HOME': 28, 'OTHER': 1, 'OTHER COMMERCIAL TRANSPORTATION': 14, 'OTHER RAILROAD PROP / TRAIN DEPOT': 54, 'PARK PROPERTY': 48, 'PARKING LOT/GARAGE(NON.RESID.)': 35, 'POLICE FACILITY/VEH PARKING LOT': 50, 'RESIDENCE': 2, 'RESIDENCE PORCH/HALLWAY': 53, 'RESIDENCE-GARAGE': 6, 'RESIDENTIAL YARD (FRONT/BACK)': 47, 'RESTAURANT': 13, 'SAVINGS AND LOAN': 38, 'SCHOOL, PRIVATE, BUILDING': 18, 'SCHOOL, PRIVATE, GROUNDS': 63, 'SCHOOL, PUBLIC, BUILDING': 32, 'SCHOOL, PUBLIC, GROUNDS': 0, 'SIDEWALK': 31, 'SMALL RETAIL STORE': 64, 'SPORTS ARENA/STADIUM': 25, 'STREET': 16, 'TAVERN/LIQUOR STORE': 17, 'TAXICAB': 45, 'VACANT LOT/LAND': 46, 'VEHICLE NON-COMMERCIAL': 12, 'VEHICLE-COMMERCIAL': 10, 'WAREHOUSE': 23},
    'Domestic': {'False': 0, 'True': 1},
    'District': {' ': 42, '1': 7, '1.0': 15, '10': 44, '10.0': 23, '11': 33, '11.0': 38, '12': 3, '12.0': 48, '13': 46, '13.0': 19, '14': 17, '14.0': 1, '15': 40, '15.0': 21, '16': 35, '16.0': 13, '17': 47, '17.0': 31, '18': 5, '18.0': 11, '19': 25, '19.0': 28, '2': 9, '2.0': 0, '20': 41, '20.0': 20, '21.0': 10, '22': 29, '22.0': 36, '23.0': 4, '24': 34, '24.0': 43, '25': 16, '25.0': 24, '3': 49, '3.0': 39, '31.0': 2, '4': 22, '4.0': 14, '5': 8, '5.0': 32, '6': 45, '6.0': 6, '7': 12, '7.0': 18, '8': 26, '8.0': 37, '9': 27, '9.0': 30},  # Add more mappings as needed
    'CommunityArea': {' ': 133, '0': 35, '0.0': 7, '1': 129, '1.0': 59, '10': 23, '10.0': 87, '11': 71, '11.0': 45, '12': 109, '12.0': 127, '13': 153, '13.0': 137, '14': 13, '14.0': 53, '15': 89, '15.0': 135, '16': 81, '16.0': 113, '17': 77, '17.0': 121, '18': 65, '18.0': 91, '19': 37, '19.0': 149, '2': 57, '2.0': 119, '20': 117, '20.0': 19, '21': 17, '21.0': 49, '22': 79, '22.0': 21, '23': 3, '23.0': 25, '24': 41, '24.0': 67, '25': 51, '25.0': 33, '26': 29, '26.0': 43, '27': 93, '27.0': 27, '28': 83, '28.0': 123, '29': 151, '29.0': 99, '3': 75, '3.0': 97, '30': 125, '30.0': 105, '31': 143, '31.0': 131, '32': 139, '32.0': 11, '33': 31, '33.0': 69, '34': 15, '34.0': 147, '35': 107, '35.0': 9, '36': 101, '36.0': 85, '37': 39, '37.0': 47, '38': 73, '38.0': 115, '39': 61, '39.0': 63, '4': 141, '4.0': 95, '40': 111, '40.0': 5, '41': 145, '41.0': 55, '42': 155, '42.0': 103, '43': 0, '43.0': 1, '44': 140, '44.0': 72, '45': 32, '45.0': 118, '46': 100, '46.0': 128, '47': 36, '47.0': 86, '48': 26, '48.0': 24, '49': 84, '49.0': 110, '5': 12, '5.0': 4, '50': 38, '50.0': 34, '51': 78, '51.0': 130, '52': 138, '52.0': 20, '53': 44, '53.0': 126, '54': 148, '54.0': 132, '55': 88, '55.0': 64, '56': 106, '56.0': 68, '57': 154, '57.0': 136, '58': 50, '58.0': 16, '59': 52, '59.0': 80, '6': 22, '6.0': 96, '60': 114, '60.0': 122, '61': 116, '61.0': 60, '62': 76, '62.0': 46, '63': 144, '63.0': 90, '64': 56, '64.0': 82, '65': 30, '65.0': 134, '66': 66, '66.0': 48, '67': 102, '67.0': 40, '68': 54, '68.0': 94, '69': 112, '69.0': 10, '7': 120, '7.0': 74, '70': 18, '70.0': 14, '71': 92, '71.0': 58, '72': 8, '72.0': 142, '73': 70, '73.0': 28, '74': 152, '74.0': 124, '75': 6, '75.0': 42, '76': 104, '76.0': 98, '77': 62, '77.0': 108, '8': 156, '8.0': 150, '9': 146, '9.0': 2},  # Add more mappings as needed
    'Year': {'2001': 11, '2002': 10, '2003': 9, '2004': 8, '2005': 7, '2006': 6, '2007': 5, '2008': 4, '2009': 3, '2010': 2, '2011': 1, '2012': 0}  # Add more mappings as needed
}



def crime_prediction(input_data):
    prediction = loaded_model.predict([input_data])
    return prediction[0]
    print(prediction)
    if (prediction[0] == 0):
        return 'The suspect has not been arrested'
    else:
        return 'The suspect has been arrested'


##############################code for the web-app#################################

def main():


    st.title('Crime Detection in Chicago')

    # getting the data
    location_description = st.selectbox('What is the Location of Crime?',
        ('STREET', 'RESIDENTIAL YARD (FRONT/BACK)', 'GAS STATION', 'PARKING LOT/GARAGE(NON.RESID.)', 'VEHICLE NON-COMMERCIAL', 'CTA GARAGE / OTHER PROPERTY', 'RESIDENCE-GARAGE', 'OTHER', 'ALLEY', 'SPORTS ARENA/STADIUM', 'VACANT LOT/LAND', 'RESIDENCE', 'SCHOOL, PUBLIC, GROUNDS', 'DRIVEWAY - RESIDENTIAL', 'POLICE FACILITY/VEH PARKING LOT', 'SIDEWALK', 'APARTMENT', 'VEHICLE-COMMERCIAL', 'AIRPORT VENDING ESTABLISHMENT', 'BAR OR TAVERN', 'PARK PROPERTY', 'HIGHWAY/EXPRESSWAY', 'COLLEGE/UNIVERSITY GROUNDS', 'SMALL RETAIL STORE', 'CAR WASH', 'FACTORY/MANUFACTURING BUILDING', 'RESTAURANT', 'FIRE STATION', 'CHA PARKING LOT/GROUNDS', 'AIRPORT EXTERIOR - NON-SECURE AREA', 'CTA TRAIN', 'GROCERY FOOD STORE', 'AIRPORT PARKING LOT', 'MOVIE HOUSE/THEATER', 'TAVERN/LIQUOR STORE', 'GOVERNMENT BUILDING/PROPERTY', 'NURSING HOME/RETIREMENT HOME', 'AIRPORT/AIRCRAFT', 'HOTEL/MOTEL', 'CHURCH/SYNAGOGUE/PLACE OF WORSHIP', 'CONSTRUCTION SITE', 'COMMERCIAL / BUSINESS OFFICE', 'SCHOOL, PUBLIC, BUILDING', 'DEPARTMENT STORE', 'WAREHOUSE', 'HOSPITAL BUILDING/GROUNDS', 'RESIDENCE PORCH/HALLWAY', 'AIRPORT EXTERIOR - SECURE AREA', 'TAXICAB', 'CHA APARTMENT', 'SCHOOL, PRIVATE, GROUNDS', 'CONVENIENCE STORE', 'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA', 'BANK', 'OTHER RAILROAD PROP / TRAIN DEPOT', 'MEDICAL/DENTAL OFFICE', 'DRUG STORE', 'NEWSSTAND', 'AIRPORT BUILDING NON-TERMINAL - SECURE AREA', 'OTHER COMMERCIAL TRANSPORTATION', 'AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA', 'LIBRARY', 'ATHLETIC CLUB', 'FOREST PRESERVE', 'BRIDGE', 'SAVINGS AND LOAN', 'DAY CARE CENTER', 'ABANDONED BUILDING', 'CURRENCY EXCHANGE', 'SCHOOL, PRIVATE, BUILDING', 'COLLEGE/UNIVERSITY RESIDENCE HALL', 'BARBERSHOP', 'BOWLING ALLEY', 'JAIL / LOCK-UP FACILITY', 'LAKEFRONT/WATERFRONT/RIVERBANK', 'APPLIANCE STORE', 'ANIMAL HOSPITAL', 'CLEANING STORE'))
    st.write('You selected:', location_description)

    domestic = st.selectbox('Is thebcriminal domestic or not?',
        ('False', 'True'))
    st.write('You selected:', domestic)

    district = st.selectbox('From which district does the criminal belong?',
                            ('6', '12', '16', '7', '2', '25', '4', '10', '9', '8', '18', '17', '1', '5', '19', '3', '15', '8.0', '24', '11', '14', '20', '22', '13', ' ', '5.0', '18.0', '13.0', '22.0', '3.0', '10.0', '25.0', '6.0', '16.0', '2.0', '9.0', '4.0', '1.0', '19.0', '15.0', '12.0', '24.0', '7.0', '11.0', '14.0', '17.0', '20.0', '31.0', '21.0', '23.0'))
    st.write('You selected:', district)

    community_area = st.selectbox('Which Year crime took place?',
                            ('69', '24', '11', '67', '35', '19', '48', '40', '29', '58', '66', '8', '70', '14', '32', '49', '7', '45', '6', '43', '63', '38', '5', '25', '76', '34', '62', '61', '17', '16', '30', '44', '18', '1', '2', '27', '39', '31', '23', '21', '28', '50', '20', '46', '64', '77', '53', '42', '52', '65', '56', '73', '68', '71', '13', '22', '4', '15', '75', '57', '12', '54', '47', '26', '3', '41', '60', '36', '37', '72', '51', '59', '10', '74', '33', '9', '55', ' ', '0', '71.0', '40.0', '22.0', '61.0', '53.0', '66.0', '24.0', '47.0', '2.0', '19.0', '46.0', '58.0', '13.0', '1.0', '25.0', '23.0', '43.0', '67.0', '70.0', '17.0', '28.0', '65.0', '75.0', '68.0', '48.0', '37.0', '56.0', '39.0', '8.0', '7.0', '30.0', '15.0', '31.0', '44.0', '18.0', '51.0', '6.0', '63.0', '60.0', '35.0', '42.0', '29.0', '73.0', '49.0', '33.0', '45.0', '21.0', '69.0', '38.0', '3.0', '54.0', '26.0', '32.0', '50.0', '59.0', '12.0', '62.0', '41.0', '16.0', '14.0', '5.0', '34.0', '11.0', '72.0', '4.0', '20.0', '77.0', '64.0', '10.0', '27.0', '55.0', '52.0', '36.0', '57.0', '9.0', '76.0', '74.0', '0.0'))
    st.write('You selected:', community_area)

    year = st.selectbox('Which year the crime occured?',
                            ('2012', '2011', '2010', '2009', '2008', '2007', '2006', '2005', '2004', '2003', '2002', '2001'))
    st.write('You selected:', year)

    # Convert selected values to their corresponding encoded numbers
    encoded_location_description = label_mappings['LocationDescription'][location_description]

    encoded_domestic = label_mappings['Domestic'][domestic]
    encoded_district = label_mappings['District'][district]
    encoded_community_area = label_mappings['CommunityArea'][community_area]
    encoded_year = label_mappings['Year'][year]



    #prediction
    if st.button('Predict'):
        # Combine encoded features into a single array
        input_features = [encoded_location_description, encoded_domestic, encoded_district,encoded_community_area, encoded_year]
        # Make predictions
        prediction = crime_prediction(input_features)

        if prediction == 0:
            st.write('Predicted result: The suspect has not been arrested')
        else:
            st.write('Predicted result: The suspect has been arrested')


if __name__ == '__main__':
    main()
