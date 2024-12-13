import psycopg2
import spacy

# Step 1: Connect to your PostgreSQL database
# Replace these values with your database credentials
db_config = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'admin',
    'host': 'localhost',  # e.g., 'localhost'
    'port': 5432          # Default PostgreSQL port
}

connection = psycopg2.connect(**db_config)
cursor = connection.cursor()

# Step 2: Query the database to get all commit messages
query = "SELECT id, message FROM commits;"  # Adjust table and column names as needed
cursor.execute(query)

# Fetch all commit messages
commit_messages = cursor.fetchall()  # Returns a list of tuples [(id1, message1), (id2, message2), ...]

# Step 3: Load the trained spaCy model
# Replace 'your_spacy_model' with the name of your trained spaCy model directory or file
nlp = spacy.load('text_categorizer_model')

ADD_FEATURE_COUNT = 0
REFACTOR_COUNT = 0

# Step 4: Analyze each commit message
results = []
for id, message in commit_messages:
    doc = nlp(message)
    # Extract the predicted category
    if doc.cats['ADD_FEATURE'] > doc.cats['REFACTOR_CODE']:
        category = 'ADD_FEATURE'
        ADD_FEATURE_COUNT += 1
    else:
        category = 'REFACTOR_CODE'
        REFACTOR_COUNT += 1
    
    # Append the result (commit_id, category) to the results list
    #results.append((category, id))

# Commit changes and close the connection
connection.commit()
cursor.close()
connection.close()




# Step 6: (Optional) Print results
#for id, category in results:
    #print(f"Commit ID: {id}, Category: {category}")

#print("refactor count: " + str(REFACTOR_COUNT) + ". new feature count: " + str(ADD_FEATURE_COUNT))
print(ADD_FEATURE_COUNT)
print(REFACTOR_COUNT)
results = [ADD_FEATURE_COUNT, REFACTOR_COUNT]