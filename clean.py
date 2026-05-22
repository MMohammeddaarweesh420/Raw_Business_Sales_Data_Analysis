import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # لتجنب مشاكل واجهات الرسوم في بعض البيئات
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime

# ==========================================
# 1. إعداد نظام التسجيل (Logging)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_pipeline.log"),
        logging.StreamHandler()
    ]
)

# ==========================================
# 2. الدوال الرئيسية (Functions)
# ==========================================

def load_and_explore_data(file_path):
    """قراءة البيانات واستكشافها"""
    try:
        logging.info(f"جاري قراءة الملف: {file_path}")
        df = pd.read_excel(file_path, engine='openpyxl')
        
        logging.info("\n--- استكشاف البيانات ---")
        logging.info(f"عدد الصفوف والأعمدة: {df.shape}")
        logging.info(f"أسماء الأعمدة:\n{df.columns.tolist()}")
        logging.info(f"أنواع البيانات:\n{df.dtypes}")
        logging.info(f"عدد القيم المفقودة:\n{df.isnull().sum()}")
        logging.info(f"ملخص إحصائي:\n{df.describe(include='all')}")
        
        return df
    except Exception as e:
        logging.error(f"خطأ في قراءة الملف: {e}")
        return None

def clean_data(df):
    """تنظيف البيانات الاحترافي"""
    logging.info("\n--- بدء تنظيف البيانات ---")
    
    # التأكد من وجود الأعمدة المطلوبة قبل المعالجة
    required_columns = [
        'Unit_Price', 'Total_Sales', 'Customer_Rating', 
        'Quantity', 'Category', 'Product', 'Date'
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"الأعمدة التالية غير موجودة: {missing_cols}")

    df_clean = df.copy()
    
    # 1. معالجة الرموز الخاصة (* و $) وتحويلها لقيم رقمية
    def clean_currency_and_asterisk(val):
        if isinstance(val, str):
            if val.strip() == '*':
                return np.nan
            return val.replace('$', '').replace(',', '').strip()
        return val

    for col in ['Unit_Price', 'Total_Sales']:
        df_clean[col] = df_clean[col].apply(clean_currency_and_asterisk)
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    df_clean['Customer_Rating'] = df_clean['Customer_Rating'].apply(lambda x: np.nan if str(x).strip() == '*' else x)
    df_clean['Customer_Rating'] = pd.to_numeric(df_clean['Customer_Rating'], errors='coerce')

    # حساب القيم المفقودة الفعلية بعد تحويل الرموز
    missing_before_imputation = df_clean.isnull().sum().sum()

    # 2. معالجة Missing Values (باستخدام الطريقة المتوافقة مع Pandas 3+)
    # Unit_Price بالـ Median
    median_price = df_clean['Unit_Price'].median()
    df_clean['Unit_Price'] = df_clean['Unit_Price'].fillna(median_price)
    
    # Total_Sales حسابها من Quantity × Unit_Price
    df_clean['Total_Sales'] = df_clean.apply(
        lambda row: row['Quantity'] * row['Unit_Price'] if pd.notnull(row['Quantity']) and pd.notnull(row['Unit_Price']) else row['Total_Sales'],
        axis=1
    )
    # إذا بقيت قيم فارغة نستخدم Median
    df_clean['Total_Sales'] = df_clean['Total_Sales'].fillna(df_clean['Total_Sales'].median())
    
    # Customer_Rating بالـ Mean
    mean_rating = df_clean['Customer_Rating'].mean()
    df_clean['Customer_Rating'] = df_clean['Customer_Rating'].fillna(round(mean_rating, 2))
    
    # 3. إزالة التكرارات
    duplicates_count = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    logging.info(f"تم حذف {duplicates_count} صف مكرر.")

    # 4. معالجة أنواع البيانات والتواريخ الفارغة
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
    df_clean['Date'] = df_clean['Date'].ffill() # منع مشاكل التواريخ الفارغة
    
    df_clean['Quantity'] = pd.to_numeric(df_clean['Quantity'], errors='coerce')

    # 5. تنظيف النصوص
    text_cols = df_clean.select_dtypes(include=['object', 'string']).columns
    for col in text_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        if col == 'Category':
            df_clean[col] = df_clean[col].str.title() # توحيد شكل الفئات

    final_missing = df_clean.isnull().sum().sum()
    logging.info(f"تمت معالجة {missing_before_imputation - final_missing} قيمة مفقودة.")
    
    return df_clean

def detect_outliers(df):
    """اكتشاف القيم الشاذة باستخدام IQR وإنشاء تقرير"""
    logging.info("\n--- اكتشاف القيم الشاذة (Outliers) ---")
    numeric_cols = ['Unit_Price', 'Quantity', 'Total_Sales', 'Customer_Rating']
    outlier_report = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_report[col] = len(outliers)
        logging.info(f"عمود {col}: يوجد {len(outliers)} قيمة شاذة")
        
    return outlier_report

def engineer_features(df):
    """هندسة الميزات الجديدة"""
    logging.info("\n--- هندسة الميزات (Feature Engineering) ---")
    
    # Sales_Per_Unit مع تجنب القسمة على صفر
    df['Sales_Per_Unit'] = np.where(
        df['Quantity'] > 0,
        df['Total_Sales'] / df['Quantity'],
        0
    )
    
    # Revenue_Level (تم تغيير default إلى نص بدل 0 لتوافق NumPy 2+)
    conditions_rev = [
        df['Total_Sales'] <= df['Total_Sales'].quantile(0.33),
        df['Total_Sales'] <= df['Total_Sales'].quantile(0.66),
        df['Total_Sales'] > df['Total_Sales'].quantile(0.66)
    ]
    choices_rev = ['Low', 'Medium', 'High']
    df['Revenue_Level'] = np.select(conditions_rev, choices_rev, default='Unknown')
    
    # Rating_Category
    conditions_rat = [
        df['Customer_Rating'] <= 2,
        df['Customer_Rating'] <= 3,
        df['Customer_Rating'] > 3
    ]
    choices_rat = ['Poor', 'Average', 'Excellent']
    df['Rating_Category'] = np.select(conditions_rat, choices_rat, default='Unknown')
    
    return df

def save_preprocessed_data(df):
    """حفظ البيانات بعد التنظيف وهندسة الميزات (قبل التجهيز النهائي لـ Power BI)"""
    logging.info("\n--- حفظ البيانات بعد المعالجة المبدئية (Preprocessed Data) ---")
    
    preprocessed_excel = "Preprocessed_Business_Sales_Data.xlsx"
    df.to_excel(preprocessed_excel, index=False, engine='openpyxl')
    
    preprocessed_csv = "Preprocessed_Business_Sales_Data.csv"
    df.to_csv(preprocessed_csv, index=False, encoding='utf-8-sig')
    
    logging.info(f"تم حفظ البيانات المعالجة مبدئياً في: {preprocessed_excel} و {preprocessed_csv}")

def perform_eda(df):
    """التحليل الاستكشافي واستخراج المؤشرات"""
    logging.info("\n--- التحليل الاستكشافي EDA ---")
    eda_results = {}
    
    eda_results['Total_Sales'] = df['Total_Sales'].sum()
    eda_results['Total_Quantity'] = df['Quantity'].sum()
    eda_results['Avg_Rating'] = round(df['Customer_Rating'].mean(), 2)
    eda_results['Top_Category'] = df.groupby('Category')['Total_Sales'].sum().idxmax()
    eda_results['Top_Product'] = df.groupby('Product')['Total_Sales'].sum().idxmax()
    eda_results['Sales_By_Category'] = df.groupby('Category')['Total_Sales'].sum()
    eda_results['Rating_Distribution'] = df['Rating_Category'].value_counts()
    eda_results['Revenue_By_Product'] = df.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False)
    
    logging.info(f"إجمالي المبيعات: ${eda_results['Total_Sales']:,.2f}")
    logging.info(f"أعلى فئة مبيعاً: {eda_results['Top_Category']}")
    logging.info(f"أفضل منتج أداءً: {eda_results['Top_Product']}")
    
    return eda_results

def create_visualizations(df, eda_results):
    """إنشاء وحفظ الرسوم البيانية"""
    logging.info("\n--- إنشاء الرسوم البيانية ---")
    vis_dir = "visualizations"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Bar Chart للمبيعات حسب الفئات
    plt.figure(figsize=(10, 6))
    cat_sales = df.groupby('Category')['Total_Sales'].sum().sort_values()
    cat_sales.plot(kind='bar', color='teal')
    plt.title('Total Sales by Category')
    plt.ylabel('Total Sales ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{vis_dir}/sales_by_category.png', dpi=300)
    plt.close()

    # 2. Pie Chart لتوزيع الفئات
    plt.figure(figsize=(8, 8))
    df['Category'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title('Category Distribution')
    plt.ylabel('')
    plt.savefig(f'{vis_dir}/category_distribution.png', dpi=300)
    plt.close()

    # 3. Line Chart للمبيعات
    plt.figure(figsize=(12, 6))
    daily_sales = df.groupby('Date')['Total_Sales'].sum()
    plt.plot(daily_sales.index, daily_sales.values, color='purple', linewidth=2)
    plt.title('Daily Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Sales ($)')
    plt.savefig(f'{vis_dir}/sales_trend.png', dpi=300)
    plt.close()

    # 4. Histogram لتوزيع التقييمات
    plt.figure(figsize=(10, 6))
    plt.hist(df['Customer_Rating'], bins=5, color='coral', edgecolor='black')
    plt.title('Customer Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.savefig(f'{vis_dir}/rating_distribution.png', dpi=300)
    plt.close()

    # 5. Heatmap للـ Correlation
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig(f'{vis_dir}/correlation_heatmap.png', dpi=300)
    plt.close()

    # 6. Top Products Chart
    plt.figure(figsize=(10, 6))
    top_products = df.groupby('Product')['Total_Sales'].sum().nlargest(10)
    top_products.sort_values().plot(kind='barh', color='skyblue')
    plt.title('Top 10 Products by Revenue')
    plt.xlabel('Total Sales ($)')
    plt.savefig(f'{vis_dir}/top_products.png', dpi=300)
    plt.close()

    logging.info(f"تم حفظ جميع الرسوم في مجلد {vis_dir}/")

def prepare_for_powerbi(df):
    """تجهيز البيانات لـ Power BI"""
    logging.info("\n--- تجهيز البيانات لـ Power BI ---")
    
    # تنظيف أسماء الأعمدة بشكل احترافي باستخدام Regex
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(' ', '_')
        .str.replace(r'[^A-Za-z0-9_]', '', regex=True)
    )
    
    # التأكد من أنواع البيانات
    df['Customer_Rating'] = df['Customer_Rating'].astype(float)
    
    # التأكد من عدم وجود قيم مفقودة نهائياً
    assert df.isnull().sum().sum() == 0, "يوجد قيم مفقودة لم تتم معالجتها!"
    
    return df

def save_outputs(df, eda_results, outlier_report):
    """حفظ الملفات النهائية والتقارير"""
    logging.info("\n--- حفظ الملفات النهائية لـ Power BI ---")
    
    # 1 & 2. حفظ Excel و CSV
    df.to_excel("Clean_Raw_Business_Sales_Data.xlsx", index=False, engine='openpyxl')
    df.to_csv("Clean_Raw_Business_Sales_Data.csv", index=False, encoding='utf-8-sig')
    
    # 3. إنشاء التقرير النصي
    report_content = f"""
=========================================================
        تقرير تحليل بيانات المبيعات (Business Insights Report)
=========================================================

[ أهم النتائج ]
- إجمالي المبيعات الكلي: ${eda_results['Total_Sales']:,.2f}
- إجمالي الكميات المباعة: {eda_results['Total_Quantity']:,.0f}
- متوسط تقييم العملاء: {eda_results['Avg_Rating']} / 5.0
- أعلى فئة منتجات مبيعاً: {eda_results['Top_Category']}
- أفضل المنتجات أداءً: {eda_results['Top_Product']}

[ مؤشرات الأداء الرئيسية (KPIs) ]
- متوسط قيمة الطلب (AOV): ${eda_results['Total_Sales']/len(df):,.2f}
- توزيع الإيرادات: 
{eda_results['Sales_By_Category'].to_string()}

[ تقرير القيم الشاذة (Outliers) - لم يتم حذفها ]
"""
    for col, count in outlier_report.items():
        report_content += f"- عمود {col}: {count} قيم شاذة\n"

    report_content += f"""

[ التوصيات ]
1. التركيز على تسويق فئة {eda_results['Top_Category']} لأنها تحقق أعلى إيرادات.
2. دراسة أسباب التقييمات المنخفضة (Poor) وتحسين جودة خدمة العملاء للمنتجات ذات الأداء الضعيف.
3. العمل على زيادة متوسط قيمة الطلب (AOV) من خلال تقديم عروض (Cross-sell) على المنتجات المكملة.

[ ملاحظات مهمة للـ Dashboarding ]
- تم توحيد أسماء الأعمدة بإزالة المسافات والرموز واستخدام (Underscore) لتجنب أخطاء DAX في Power BI.
- تم الاحتفاظ بنوع بيانات التاريخ كـ DateTime لتسهيل Time Intelligence في Power BI.
- لم يتم حذف القيم الشاذة، ولكن تم تسجيلها، يرجى الانتباه لها عند تصميم المخططات.
=========================================================
"""
    with open("Business_Insights_Report.txt", "w", encoding="utf-8") as f:
        f.write(report_content)

# ==========================================
# 3. التشغيل الرئيسي (Main Execution)
# ==========================================
if __name__ == "__main__":
    FILE_NAME = "Raw_Business_Sales_Data.xlsx"
    
    try:
        start_time = datetime.now()
        logging.info("🚀 بدء تشغيل خط معالجة البيانات...")
        
        # أولاً: قراءة البيانات
        df_raw = load_and_explore_data(FILE_NAME)
        if df_raw is None:
            raise ValueError("فشل في تحميل البيانات!")
            
        rows_before = len(df_raw)
        
        # ثانياً: تنظيف البيانات
        df_clean = clean_data(df_raw)
        
        # معالجة القيم الشاذة (الاكتشاف فقط)
        outlier_report = detect_outliers(df_clean)
        
        # ثالثاً: Feature Engineering
        df_featured = engineer_features(df_clean)
        
        # رابعاً: حفظ البيانات بعد المعالجة المبدئية
        save_preprocessed_data(df_featured)
        
        # خامساً: التحليل الاستكشافي
        eda_results = perform_eda(df_featured)
        
        # سادساً: الرسوم البيانية
        create_visualizations(df_featured, eda_results)
        
        # سابعاً: تجهيز لـ Power BI
        df_pbi = prepare_for_powerbi(df_featured)
        
        # ثامناً: حفظ الملفات النهائية
        save_outputs(df_pbi, eda_results, outlier_report)
        
        rows_after = len(df_pbi)
        end_time = datetime.now()
        
        # تاسعاً: مخرجات التنفيذ
        print("\n" + "="*50)
        print("🎉 تمت عملية التحليل والتجهيز بنجاح! 🎉")
        print("="*50)
        print(f"🕒 مدة التنفيذ: {(end_time - start_time).total_seconds():.2f} ثانية")
        print(f"📊 عدد الصفوف قبل التنظيف: {rows_before}")
        print(f"📊 عدد الصفوف بعد التنظيف: {rows_after}")
        print(f"🧹 إجمالي القيم المفقودة التي تمت معالجتها: {df_raw.isnull().sum().sum() + (df_raw.applymap(lambda x: str(x).strip() == '*').sum().sum() if hasattr(df_raw, 'applymap') else 0)}")
        print("💾 أماكن حفظ الملفات:")
        print("   1. [Preprocessed] Preprocessed_Business_Sales_Data.xlsx / .csv")
        print("   2. [Power BI] Clean_Raw_Business_Sales_Data.xlsx / .csv")
        print("   3. [Report] Business_Insights_Report.txt")
        print("   4. [Charts] visualizations/ (مجلد يحتوي على 6 رسوم بيانية)")
        print("="*50)
        
    except Exception as e:
        logging.critical(f"❌ حدث خطأ فادح أثناء التنفيذ: {e}")
