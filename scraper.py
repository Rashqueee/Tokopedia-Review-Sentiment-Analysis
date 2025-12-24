from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium. webdriver.support import expected_conditions as EC
import time
import pandas as pd
import re

def scrap_reviews_by_rating(rating_filter=None):    
    url = "https://www.tokopedia.com/ismile-indonesia/apple-iphone-13-garansi-resmi-128gb-256gb-512gb-1733645446355912587/review"
    
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    # options.add_argument("--headless")
    
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    
    wait = WebDriverWait(driver, 20)
    
    print(f"\n{'='*60}")
    print(f"Scraping Review dengan Rating:  {rating_filter if rating_filter else 'SEMUA'}")
    print(f"{'='*60}")
    
    # Tunggu halaman load
    time.sleep(5)
    
    # === KLIK CHECKBOX FILTER RATING ===
    if rating_filter:
        try:
            print(f"Mencari checkbox rating {rating_filter}...")
            
            # Scroll ke area filter
            driver.execute_script("window.scrollTo(0, 300);")
            time.sleep(2)
            
            # Cari checkbox berdasarkan data-testid dan bintang            
            checkbox_xpath = f"//div[@data-testid='ratingFilter']//span[contains(text(), '{rating_filter}')]/ancestor::label//input[@type='checkbox']"
            
            try:
                checkbox = driver.find_element(By.XPATH, checkbox_xpath)
            except:
                rating_index = 5 - rating_filter 
                checkbox_xpath_alt = f"(//div[@data-testid='ratingFilter']//input[@type='checkbox'])[{rating_index + 1}]"
                checkbox = driver.find_element(By.XPATH, checkbox_xpath_alt)
            
            # Cek apakah sudah checked
            is_checked = checkbox.get_attribute("aria-checked")
            
            if is_checked != "true":
                # Klik parent label atau div untuk toggle checkbox
                parent = checkbox.find_element(By.XPATH, "./..")
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", parent)
                time.sleep(1)
                
                try:
                    parent.click()
                except:
                    driver.execute_script("arguments[0].click();", parent)
                
                print(f"Checkbox rating {rating_filter} diklik")
                time.sleep(3)
            else:
                print(f"Checkbox rating {rating_filter} sudah aktif")
                
        except Exception as e:
            print(f"Error saat mengaktifkan filter rating {rating_filter}: {e}")
            driver.save_screenshot(f"error_filter_rating_{rating_filter}.png")
            driver.quit()
            return []
    
    # === SCRAPING REVIEWS ===
    data = []
    page = 1
    max_retries = 3
    
    while True:
        print(f"\nHalaman {page}...")
        
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='lblItemUlasan']")))
        except:
            print("Timeout - tidak ada review ditemukan")
            break
        
        # Scroll dan expand
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        
        # Klik tombol "Selengkapnya"
        print("  Mengexpand review...")
        for attempt in range(5):
            try:
                selengkapnya_buttons = driver.find_elements(
                    By.XPATH, 
                    "//button[normalize-space(text())='Selengkapnya']"
                )
                
                if len(selengkapnya_buttons) == 0:
                    break
                
                clicked_count = 0
                for button in selengkapnya_buttons:
                    try:
                        if button.text.strip() == "Selengkapnya" and button.is_displayed():
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                            time. sleep(0.2)
                            button.click()
                            clicked_count += 1
                            time.sleep(0.1)
                    except: 
                        continue
                
                print(f"    {clicked_count} review di-expand")
                if clicked_count == 0:
                    break
                    
            except:
                break
        
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        
        # Parse HTML
        soup = BeautifulSoup(driver.page_source, "html.parser")
        reviews = soup.find_all('span', attrs={'data-testid': 'lblItemUlasan'})
        ratings = soup.find_all('div', attrs={'data-testid': 'icnStarRating'})
        
        print(f"  {len(reviews)} review ditemukan di halaman ini")
        
        if len(reviews) == 0:
            print("  Tidak ada review, stop scraping")
            break
        
        # Extract data
        page_count = 0
        for idx, review_span in enumerate(reviews):
            try:
                review_text = review_span.text.strip()
                review_text = review_text.replace("Selengkapnya", "").replace("Tutup Ulasan", "").strip()
                
                rating = None
                if idx < len(ratings):
                    rating_label = ratings[idx].get('aria-label', '')
                    rating_match = re.search(r'(\d+)', rating_label)
                    if rating_match:
                        rating = int(rating_match.group(1))
                
                if review_text:
                    data.append({
                        "review": review_text,
                        "rating": rating,
                    })
                    page_count += 1
                    
            except Exception as e:
                continue
        
        print(f"  {page_count} review ditambahkan | Total: {len(data)}")
        
        # Next page
        has_next = False
        for retry in range(max_retries):
            try:
                next_button = wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button[aria-label*='berikutnya']")
                ))
                
                if next_button.get_attribute("disabled"):
                    print("  Tombol next disabled")
                    break
                
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
                time.sleep(1)
                
                try:
                    next_button.click()
                except:
                    driver.execute_script("arguments[0].click();", next_button)
                
                print("  Navigasi ke halaman berikutnya...")
                time.sleep(5)
                has_next = True
                page += 1
                break
                
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"  Tidak ada halaman berikutnya")
                    has_next = False
        
        if not has_next:
            break
    
    driver.quit()
    print(f"\nSelesai scraping rating {rating_filter}:  {len(data)} review")
    return data


# === MAIN EXECUTION ===
print("="*60)
print("TOKOPEDIA REVIEW SCRAPER - MULTI RATING FILTER")
print("="*60)

all_data = []

# Scrap untuk setiap rating (dari 1 hingga 5)
for rating in [1, 2, 3, 4, 5]:
    print(f"\n{'🌟'} RATING {rating} {'🌟'}")
    rating_data = scrap_reviews_by_rating(rating)
    
    if len(rating_data) > 0:
        all_data.extend(rating_data)
        print(f"Berhasil:  {len(rating_data)} review rating {rating}")
    else:
        print(f"Tidak ada review dengan rating {rating}")
    
    time.sleep(3)

# === SAVE TO CSV ===
print(f"\n{'='*60}")
print("Menyimpan data...")
print(f"{'='*60}")

if len(all_data) > 0:
    df = pd.DataFrame(all_data)
    df_unique = df.drop_duplicates(subset=['review'])
    
    df_unique.to_csv("data/raw/tokopedia_reviews.csv", index=False, encoding='utf-8-sig')
    
    print(f"\nSELESAI!")
    print(f"  • Total review:  {len(df_unique)}")
    print(f"  • Duplikat dihapus: {len(df) - len(df_unique)}")
    print(f"  • File:  data/raw/tokopedia_reviews.csv")
    
    # Statistik per rating
    print(f"\nDistribusi Rating:")
    rating_dist = df_unique['rating'].value_counts().sort_index()
    
    for r in range(1, 6):
        count = rating_dist.get(r, 0)
        if len(df_unique) > 0:
            percentage = (count / len(df_unique)) * 100
            bar = "█" * int(percentage / 2)
            print(f"  ⭐ {r}: {bar} {count: 4d} ({percentage: 5.1f}%)")
        else:
            print(f"  ⭐ {r}: 0")
    
    # Analisis
    print(f"\nAnalisis:")
    if 1 not in rating_dist and 2 not in rating_dist:
        print(f"   Review rating 1 dan 2 TIDAK DITEMUKAN")
        print(f"   Kemungkinan produk ini sangat berkualitas")
    else:
        print(f"   Data lengkap dari rating 1-5")
        
else:
    print("\nGAGAL:  Tidak ada data yang berhasil di-scrape!")