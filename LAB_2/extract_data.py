import json

PATH = 'works.json'

# Load the input JSON file
with open(PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare a list to hold the extracted data
extracted_articles = []

# Iterate through the articles and extract the necessary information
for article in data['results']:
    # Extract title
    title = article.get('title', False)

    # Extract authors
    authors = [authorship['author']['display_name'] for authorship in article.get('authorships', False)]

    if authors == [] or not authors:
        authors = False

    print(authors)

    # Extract abstract (if it's stored as an inverted index, convert it to normal text)
    abstract_index = article.get('abstract_inverted_index', False)
    if abstract_index:
        # Create a list with (index, word) pairs, then sort by index
        abstract_items = []
        for word, positions in abstract_index.items():
            for pos in positions:
                abstract_items.append((pos, word))
        
        # Sort the items by index to reconstruct the abstract
        abstract_items.sort(key=lambda x: x[0])
        abstract = ' '.join([word for _, word in abstract_items]).replace('\n', ' ').replace('\r', ' ')
    else:
        abstract = False

    # Extract keywords
    keywords = [keyword['display_name'] for keyword in article.get('keywords', False)]

    # Add extracted data to the list if all of the fields have some content
    if (title and authors and abstract and keywords) and len(extracted_articles) < 30:
        extracted_articles.append({
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'keywords': keywords
        })
    else:
        print(f"Skipping article with title: {title}")

# Write the extracted data to a new JSON file
with open('extracted_articles.json', 'w', encoding='utf-8') as f:
    json.dump(extracted_articles, f, ensure_ascii=False, indent=4)

print("Extracted data has been saved to extracted_articles.json.")
