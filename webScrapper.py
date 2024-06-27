import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# Set up the WebDriver with automatic driver management
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

reviews = []
base_url = 'https://ca.trustpilot.com/review/www.lululemon.com?page='

for page in range(1, 47):  # Loop through pages 1 to 46
    url = f'{base_url}{page}'
    driver.get(url)
    time.sleep(2)  # Wait for page to load

    # Extract JSON-LD content
    json_ld_elements = driver.find_elements(By.CSS_SELECTOR, 'script[type="application/ld+json"][data-business-unit-json-ld="true"]')
    for element in json_ld_elements:
        json_ld = element.get_attribute('innerHTML')
        data = json.loads(json_ld)
        for item in data['@graph']:
            if item['@type'] == 'Review':
                review_text = item.get('reviewBody', '')  # Extract only the review text
                if review_text:
                    reviews.append(review_text)

driver.quit()

# Save reviews to a CSV file
df = pd.DataFrame(reviews, columns=['Review'])
df.to_csv('lululemon_reviews.csv', index=False)

print(f"Scraped {len(reviews)} reviews.")
