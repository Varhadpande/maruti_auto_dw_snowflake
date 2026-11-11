import os
import random
import string
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# CONFIG – adjust these numbers to control volume
# ============================================================

BASE_PATH = "./output_raw"          # where CSVs will be written
MAX_BYTES_PER_FILE = 16 * 1024 * 1024   # 16 MB split threshold

# Core volumes
N_PLANTS = 10
N_VENDORS = 120
N_MATERIALS = 400
N_BRANDS = 3
N_MODELS = 20
N_VARIANTS = 60
N_DEALERS = 70                 # between 50–80
SHOWROOMS_PER_DEALER = (2, 3)  # min, max
SERVICE_CENTERS_PER_DEALER = (1, 1)

# 👉 Increase this to 15000 to meet “15–17k customers per dealer”:
CUSTOMERS_PER_DEALER = 3000    # adjust as needed
N_SALESPERSON_PER_DEALER = 6

# Transaction volumes (tune as you like)
N_CONTRACTS = 4000
N_PO_HEADER = 60000
N_GRN = 60000
N_VEHICLE_PRODUCTION = 120000
N_PRODUCTION_DAILY = 60000
N_PLANT_INV = 60000
N_DEALER_STOCK = 80000
N_DISPATCH = 60000
N_BOOKING = 120000
N_INVOICE = 100000
N_SERVICE_ORDERS = 80000
N_SERVICE_LINES = 120000
N_WARRANTY = 20000
N_FEEDBACK = 60000
N_FORECAST = 80000
N_BACKLOG = 60000

# Randomness
np.random.seed(42)
random.seed(42)

# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def rand_code(prefix, n, width=3):
    return [f"{prefix}{i:0{width}d}" for i in range(1, n + 1)]


def random_string(length=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def write_csv_split(df: pd.DataFrame, base_name: str):
    """
    Write df to one or more CSVs under BASE_PATH, each < MAX_BYTES_PER_FILE.
    Returns list of filenames.
    """
    ensure_dir(BASE_PATH)
    temp_path = os.path.join(BASE_PATH, f"{base_name}_temp.csv")
    df.to_csv(temp_path, index=False)
    size = os.path.getsize(temp_path)

    if size <= MAX_BYTES_PER_FILE:
        final_path = os.path.join(BASE_PATH, f"{base_name}.csv")
        os.replace(temp_path, final_path)
        print(f"[OK] {final_path} ({size/1024/1024:.2f} MB)")
        return [final_path]

    # Need splitting
    os.remove(temp_path)
    parts = int(size // MAX_BYTES_PER_FILE) + 1
    chunk_size = len(df) // parts + 1
    files = []
    for i in range(parts):
        chunk = df.iloc[i*chunk_size:(i+1)*chunk_size]
        if chunk.empty:
            continue
        fname = os.path.join(BASE_PATH, f"{base_name}_part{i+1}.csv")
        chunk.to_csv(fname, index=False)
        f_size = os.path.getsize(fname)
        print(f"[OK] {fname} ({f_size/1024/1024:.2f} MB)")
        files.append(fname)
    return files


# Geography scaffolding
regions = ["North", "South", "East", "West", "Central"]
states_by_region = {
    "North": ["Delhi", "Punjab", "Haryana"],
    "South": ["Karnataka", "Tamil Nadu", "Telangana"],
    "East": ["West Bengal", "Odisha"],
    "West": ["Maharashtra", "Gujarat", "Rajasthan"],
    "Central": ["Madhya Pradesh", "Uttar Pradesh"]
}
cities_by_state = {
    "Delhi": ["Delhi"],
    "Punjab": ["Ludhiana", "Amritsar"],
    "Haryana": ["Gurgaon", "Faridabad"],
    "Karnataka": ["Bengaluru", "Mysuru"],
    "Tamil Nadu": ["Chennai", "Coimbatore"],
    "Telangana": ["Hyderabad"],
    "West Bengal": ["Kolkata"],
    "Odisha": ["Bhubaneswar"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
    "Gujarat": ["Ahmedabad", "Surat"],
    "Rajasthan": ["Jaipur", "Udaipur"],
    "Madhya Pradesh": ["Bhopal", "Indore"],
    "Uttar Pradesh": ["Lucknow", "Noida"]
}


def random_geo(n: int):
    rows = []
    for _ in range(n):
        region = random.choice(regions)
        state = random.choice(states_by_region[region])
        city = random.choice(cities_by_state[state])
        pin_code = str(random.randint(100000, 999999))
        rows.append((region, state, city, pin_code, "India"))
    return rows


# Global date range 2023–2025
DATE_RANGE = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")


# ============================================================
# MAIN GENERATOR
# ============================================================

def main():
    batch_id = "BATCH_001"
    now_ts = datetime.utcnow()

    # --------------------------------------------------------
    # 1. MASTER DATA
    # --------------------------------------------------------

    # 1.1 PLANT
    plant_ids = [f"P{i:03d}" for i in range(1, N_PLANTS + 1)]
    plant_geo = random_geo(N_PLANTS)
    erp_plant = pd.DataFrame({
        "PLANT_ID": plant_ids,
        "PLANT_CODE": rand_code("PL", N_PLANTS),
        "PLANT_NAME": [f"Maruti Plant {i}" for i in range(1, N_PLANTS + 1)],
        "PLANT_TYPE": np.random.choice(["Assembly", "Engine", "Body"], N_PLANTS),
        "COMPANY_CODE": "MSIL",
        "ADDRESS_LINE1": "Address Line 1",
        "ADDRESS_LINE2": "Address Line 2",
        "CITY": [c[2] for c in plant_geo],
        "DISTRICT": [c[2] for c in plant_geo],
        "STATE": [c[1] for c in plant_geo],
        "COUNTRY": [c[4] for c in plant_geo],
        "PIN_CODE": [c[3] for c in plant_geo],
        "REGION": [c[0] for c in plant_geo],
        "START_DATE": "2010-01-01",
        "END_DATE": "",
        "ACTIVE_FLAG": "Y",
        "DEFAULT_SHIFT_PATTERN": "3",
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_PLANT.csv"
    })
    write_csv_split(erp_plant, "RAW_ERP_PLANT")

    # 1.2 VENDOR
    vendor_ids = [f"V{i:05d}" for i in range(1, N_VENDORS + 1)]
    vend_geo = random_geo(N_VENDORS)
    erp_vendor = pd.DataFrame({
        "VENDOR_ID": vendor_ids,
        "VENDOR_CODE": rand_code("VD", N_VENDORS, width=4),
        "VENDOR_NAME": [f"Vendor {i}" for i in range(1, N_VENDORS + 1)],
        "VENDOR_TYPE": np.random.choice(["Tier1", "Tier2"], N_VENDORS),
        "ADDRESS_LINE1": "Vendor Address1",
        "ADDRESS_LINE2": "Vendor Address2",
        "CITY": [c[2] for c in vend_geo],
        "DISTRICT": [c[2] for c in vend_geo],
        "STATE": [c[1] for c in vend_geo],
        "COUNTRY": [c[4] for c in vend_geo],
        "PIN_CODE": [c[3] for c in vend_geo],
        "REGION": [c[0] for c in vend_geo],
        "CONTACT_PERSON": [f"Contact {i}" for i in range(1, N_VENDORS + 1)],
        "CONTACT_EMAIL": [f"vendor{i}@example.com" for i in range(1, N_VENDORS + 1)],
        "CONTACT_PHONE": [f"9{random.randint(10**9,10**10-1)}" for _ in range(N_VENDORS)],
        "DEFAULT_CURRENCY": "INR",
        "PAYMENT_TERMS_CODE": "NET30",
        "PREFERRED_FLAG": np.random.choice(["Y", "N"], N_VENDORS, p=[0.3, 0.7]),
        "AVG_HIST_LEAD_TIME_DAYS": np.random.randint(5, 30, N_VENDORS).astype(str),
        "LEAD_TIME_VARIABILITY_DAYS": np.random.randint(1, 10, N_VENDORS).astype(str),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_VENDOR.csv"
    })
    write_csv_split(erp_vendor, "RAW_ERP_VENDOR")

    # 1.3 MATERIAL
    material_ids = [f"M{i:06d}" for i in range(1, N_MATERIALS + 1)]
    erp_material = pd.DataFrame({
        "MATERIAL_ID": material_ids,
        "MATERIAL_CODE": rand_code("MAT", N_MATERIALS, width=5),
        "MATERIAL_NAME": [f"Material {i}" for i in range(1, N_MATERIALS + 1)],
        "MATERIAL_CATEGORY": np.random.choice(["Steel", "Plastic", "Electronics", "Glass"], N_MATERIALS),
        "MATERIAL_GROUP": np.random.choice(["BODY", "ENGINE", "INTERIOR"], N_MATERIALS),
        "BASE_UOM": np.random.choice(["KG", "PCS"], N_MATERIALS),
        "PURCHASE_UOM": np.random.choice(["KG", "PCS"], N_MATERIALS),
        "CRITICALITY_LEVEL": np.random.choice(["HIGH", "MEDIUM", "LOW"], N_MATERIALS),
        "HAZARDOUS_FLAG": np.random.choice(["Y", "N"], N_MATERIALS, p=[0.1, 0.9]),
        "USED_IN_BOM_FLAG": np.random.choice(["Y", "N"], N_MATERIALS, p=[0.8, 0.2]),
        "STATUS": "ACTIVE",
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_MATERIAL.csv"
    })
    write_csv_split(erp_material, "RAW_ERP_MATERIAL")

    # 1.4 BRAND / MODEL / VARIANT
    brand_ids = [f"B{i:03d}" for i in range(1, N_BRANDS + 1)]
    erp_brand = pd.DataFrame({
        "BRAND_ID": brand_ids,
        "BRAND_CODE": rand_code("BR", N_BRANDS),
        "BRAND_NAME": [f"Maruti Brand {i}" for i in range(1, N_BRANDS + 1)],
        "ACTIVE_FLAG": "Y",
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_BRAND.csv"
    })
    write_csv_split(erp_brand, "RAW_ERP_BRAND")

    model_ids = [f"MD{i:03d}" for i in range(1, N_MODELS + 1)]
    model_brand_ids = np.random.choice(brand_ids, N_MODELS)
    erp_model = pd.DataFrame({
        "MODEL_ID": model_ids,
        "BRAND_ID": model_brand_ids,
        "MODEL_CODE": rand_code("MDL", N_MODELS),
        "MODEL_NAME": [f"Model {i}" for i in range(1, N_MODELS + 1)],
        "BODY_TYPE": np.random.choice(["Hatchback", "Sedan", "SUV", "MPV"], N_MODELS),
        "LAUNCH_DATE": np.random.choice(DATE_RANGE[:365]).astype(str),
        "DISCONTINUED_DATE": "",
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_MODEL.csv"
    })
    write_csv_split(erp_model, "RAW_ERP_MODEL")

    variant_ids = [f"VRT{i:03d}" for i in range(1, N_VARIANTS + 1)]
    variant_model_ids = np.random.choice(model_ids, N_VARIANTS)
    erp_variant = pd.DataFrame({
        "VARIANT_ID": variant_ids,
        "MODEL_ID": variant_model_ids,
        "VARIANT_CODE": rand_code("VAR", N_VARIANTS),
        "VARIANT_NAME": [f"Variant {i}" for i in range(1, N_VARIANTS + 1)],
        "FUEL_TYPE": np.random.choice(["Petrol", "Diesel", "CNG", "Hybrid"], N_VARIANTS),
        "TRANSMISSION_TYPE": np.random.choice(["MT", "AT", "AMT"], N_VARIANTS),
        "ENGINE_CC": np.random.randint(800, 2000, N_VARIANTS).astype(str),
        "TRIM_LEVEL": np.random.choice(["Base", "Mid", "Top"], N_VARIANTS),
        "ACTIVE_FLAG": "Y",
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_VARIANT.csv"
    })
    write_csv_split(erp_variant, "RAW_ERP_VARIANT")

    # 1.5 DEALER / SHOWROOM / SERVICE CENTER
    dealer_ids = [f"D{i:04d}" for i in range(1, N_DEALERS + 1)]
    dealer_geo = random_geo(N_DEALERS)
    erp_dealer = pd.DataFrame({
        "DEALER_ID": dealer_ids,
        "DEALER_CODE": rand_code("DLR", N_DEALERS, width=4),
        "DEALER_GROUP_NAME": [f"Dealer Group {i % 10 + 1}" for i in range(1, N_DEALERS + 1)],
        "GSTIN": [random_string(15) for _ in range(N_DEALERS)],
        "PAN_NO": [random_string(10) for _ in range(N_DEALERS)],
        "ADDRESS_LINE1": "Dealer Address1",
        "ADDRESS_LINE2": "Dealer Address2",
        "CITY": [c[2] for c in dealer_geo],
        "DISTRICT": [c[2] for c in dealer_geo],
        "STATE": [c[1] for c in dealer_geo],
        "COUNTRY": [c[4] for c in dealer_geo],
        "PIN_CODE": [c[3] for c in dealer_geo],
        "REGION": [c[0] for c in dealer_geo],
        "CHANNEL": np.random.choice(["NEXA", "Arena", "Rural"], N_DEALERS),
        "OPENING_DATE": "2015-01-01",
        "CLOSING_DATE": "",
        "ACTIVE_FLAG": "Y",
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_DEALER.csv"
    })
    write_csv_split(erp_dealer, "RAW_ERP_DEALER")

    # SHOWROOM
    showroom_rows = []
    sr_id_seq = 1
    for d_id in dealer_ids:
        n_sr = random.randint(*SHOWROOMS_PER_DEALER)
        for _ in range(n_sr):
            geo = random.choice(dealer_geo)
            showroom_rows.append({
                "SHOWROOM_ID": f"SR{sr_id_seq:05d}",
                "DEALER_ID": d_id,
                "SHOWROOM_CODE": f"SR{random.randint(1000,9999)}",
                "SHOWROOM_NAME": f"Showroom {sr_id_seq}",
                "SHOWROOM_TYPE": random.choice(["Main", "Outlet"]),
                "ADDRESS_LINE1": "Showroom Address1",
                "ADDRESS_LINE2": "Showroom Address2",
                "CITY": geo[2],
                "DISTRICT": geo[2],
                "STATE": geo[1],
                "COUNTRY": geo[4],
                "PIN_CODE": geo[3],
                "CHANNEL": random.choice(["NEXA", "Arena", "Rural"]),
                "OPENING_DATE": "2016-01-01",
                "CLOSING_DATE": "",
                "ACTIVE_FLAG": "Y",
                "HAS_SERVICE_CENTER_FLAG": random.choice(["Y", "N"]),
                "DW_CREATED_TS": now_ts,
                "DW_LAST_UPDATED_TS": now_ts,
                "DW_BATCH_ID": batch_id,
                "DW_SOURCE_FILE_NAME": "ERP_SHOWROOM.csv"
            })
            sr_id_seq += 1
    erp_showroom = pd.DataFrame(showroom_rows)
    write_csv_split(erp_showroom, "RAW_ERP_SHOWROOM")

    # SERVICE CENTER
    sc_rows = []
    sc_id_seq = 1
    for d_id in dealer_ids:
        n_sc = random.randint(*SERVICE_CENTERS_PER_DEALER)
        for _ in range(n_sc):
            geo = random.choice(dealer_geo)
            sc_rows.append({
                "SERVICE_CENTER_ID": f"SC{sc_id_seq:05d}",
                "DEALER_ID": d_id,
                "SERVICE_CENTER_CODE": f"SC{random.randint(1000,9999)}",
                "SERVICE_CENTER_NAME": f"Service Center {sc_id_seq}",
                "ADDRESS_LINE1": "Service Address1",
                "ADDRESS_LINE2": "Service Address2",
                "CITY": geo[2],
                "DISTRICT": geo[2],
                "STATE": geo[1],
                "COUNTRY": geo[4],
                "PIN_CODE": geo[3],
                "IS_AUTHORIZED_FLAG": "Y",
                "OPENING_DATE": "2016-06-01",
                "ACTIVE_FLAG": "Y",
                "DW_CREATED_TS": now_ts,
                "DW_LAST_UPDATED_TS": now_ts,
                "DW_BATCH_ID": batch_id,
                "DW_SOURCE_FILE_NAME": "ERP_SERVICE_CENTER.csv"
            })
            sc_id_seq += 1
    erp_service_center = pd.DataFrame(sc_rows)
    write_csv_split(erp_service_center, "RAW_ERP_SERVICE_CENTER")

    # 1.6 CUSTOMER & SALESPERSON
    N_CUSTOMERS = N_DEALERS * CUSTOMERS_PER_DEALER
    cust_ids = [f"CUST{i:07d}" for i in range(1, N_CUSTOMERS + 1)]
    cust_geo = random_geo(N_CUSTOMERS)
    genders = ["M", "F", "O"]
    erp_customer = pd.DataFrame({
        "CUSTOMER_ID": cust_ids,
        "CUSTOMER_EXTERNAL_ID": [f"CRM{i:07d}" for i in range(1, N_CUSTOMERS + 1)],
        "CUSTOMER_NAME": [f"Customer {i}" for i in range(1, N_CUSTOMERS + 1)],
        "GENDER": np.random.choice(genders, N_CUSTOMERS, p=[0.48, 0.48, 0.04]),
        "DOB": np.random.choice(pd.date_range("1960-01-01", "2004-12-31"), N_CUSTOMERS).astype(str),
        "MOBILE_NUMBER": [f"9{random.randint(10**9,10**10-1)}" for _ in range(N_CUSTOMERS)],
        "EMAIL_ID": [f"customer{i}@example.com" for i in range(1, N_CUSTOMERS + 1)],
        "ADDRESS_LINE1": "Cust Addr1",
        "ADDRESS_LINE2": "Cust Addr2",
        "CITY": [g[2] for g in cust_geo],
        "DISTRICT": [g[2] for g in cust_geo],
        "STATE": [g[1] for g in cust_geo],
        "COUNTRY": [g[4] for g in cust_geo],
        "PIN_CODE": [g[3] for g in cust_geo],
        "CUSTOMER_SEGMENT": np.random.choice(["Retail", "Fleet", "Corporate"], N_CUSTOMERS, p=[0.7, 0.15, 0.15]),
        "LOYALTY_TIER": np.random.choice(["Silver", "Gold", "Platinum", "None"], N_CUSTOMERS, p=[0.3, 0.2, 0.1, 0.4]),
        "FIRST_PURCHASE_DATE": np.random.choice(DATE_RANGE, N_CUSTOMERS).astype(str),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_CUSTOMER.csv"
    })
    write_csv_split(erp_customer, "RAW_ERP_CUSTOMER")

    # SALESPERSON
    sp_rows = []
    sp_id_seq = 1
    for d_id in dealer_ids:
        for _ in range(N_SALESPERSON_PER_DEALER):
            sp_rows.append({
                "SALESPERSON_ID": f"SP{sp_id_seq:06d}",
                "DEALER_ID": d_id,
                "EMPLOYEE_NO": f"E{sp_id_seq:06d}",
                "FULL_NAME": f"Salesperson {sp_id_seq}",
                "DESIGNATION": random.choice(["Sales Executive", "Team Lead", "Manager"]),
                "DEPARTMENT": "Sales",
                "JOIN_DATE": "2018-01-01",
                "EXIT_DATE": "",
                "ACTIVE_FLAG": "Y",
                "DW_CREATED_TS": now_ts,
                "DW_LAST_UPDATED_TS": now_ts,
                "DW_BATCH_ID": batch_id,
                "DW_SOURCE_FILE_NAME": "ERP_SALESPERSON.csv"
            })
            sp_id_seq += 1
    erp_salesperson = pd.DataFrame(sp_rows)
    write_csv_split(erp_salesperson, "RAW_ERP_SALESPERSON")

    # --------------------------------------------------------
    # 2. PROCUREMENT & SUPPLY
    # --------------------------------------------------------

    # 2.1 PURCHASE CONTRACT HEADER & LINES
    contract_ids = [f"CT{i:06d}" for i in range(1, N_CONTRACTS + 1)]
    erp_purchase_contract = pd.DataFrame({
        "CONTRACT_ID": contract_ids,
        "CONTRACT_NO": [f"CN{random.randint(100000,999999)}" for _ in range(N_CONTRACTS)],
        "VENDOR_ID": np.random.choice(vendor_ids, N_CONTRACTS),
        "PLANT_ID": np.random.choice(plant_ids, N_CONTRACTS),
        "VALID_FROM_DATE": np.random.choice(DATE_RANGE[:365]).astype(str),
        "VALID_TO_DATE": np.random.choice(DATE_RANGE[365:730]).astype(str),
        "PAYMENT_TERMS_CODE": "NET30",
        "STANDARD_LEAD_TIME_DAYS": np.random.randint(5, 30, N_CONTRACTS).astype(str),
        "LEAD_TIME_BUFFER_DAYS": np.random.randint(1, 7, N_CONTRACTS).astype(str),
        "CONTRACT_STATUS": np.random.choice(["ACTIVE", "EXPIRED"], N_CONTRACTS, p=[0.8, 0.2]),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_PURCHASE_CONTRACT.csv"
    })
    write_csv_split(erp_purchase_contract, "RAW_ERP_PURCHASE_CONTRACT")

    contract_line_rows = []
    cl_id_seq = 1
    for c_id in contract_ids:
        n_lines = random.randint(1, 5)
        for _ in range(n_lines):
            contract_line_rows.append({
                "CONTRACT_LINE_ID": f"CL{cl_id_seq:06d}",
                "CONTRACT_ID": c_id,
                "MATERIAL_ID": random.choice(material_ids),
                "MIN_ORDER_QTY": str(random.randint(100, 5000)),
                "PRICE_PER_UOM": str(round(random.uniform(50, 2000), 2)),
                "PRICE_CURRENCY": "INR",
                "PENALTY_LATE_DELIVERY_FLAG": random.choice(["Y", "N"]),
                "DW_CREATED_TS": now_ts,
                "DW_LAST_UPDATED_TS": now_ts,
                "DW_BATCH_ID": batch_id,
                "DW_SOURCE_FILE_NAME": "ERP_PURCHASE_CONTRACT_LINE.csv"
            })
            cl_id_seq += 1
    erp_purchase_contract_line = pd.DataFrame(contract_line_rows)
    write_csv_split(erp_purchase_contract_line, "RAW_ERP_PURCHASE_CONTRACT_LINE")

    # 2.2 PO HEADER & LINES
    po_ids = [f"PO{i:08d}" for i in range(1, N_PO_HEADER + 1)]
    po_dates = np.random.choice(DATE_RANGE, N_PO_HEADER)
    erp_po_header = pd.DataFrame({
        "PO_ID": po_ids,
        "PO_NO": [f"PO{random.randint(100000,999999)}" for _ in range(N_PO_HEADER)],
        "PO_DATE": po_dates.astype(str),
        "VENDOR_ID": np.random.choice(vendor_ids, N_PO_HEADER),
        "PLANT_ID": np.random.choice(plant_ids, N_PO_HEADER),
        "CONTRACT_ID": np.random.choice(contract_ids, N_PO_HEADER),
        "CURRENCY_CODE": "INR",
        "PAYMENT_TERMS_CODE": "NET30",
        "PO_STATUS": np.random.choice(["OPEN", "PARTIALLY_RECEIVED", "CLOSED"], N_PO_HEADER, p=[0.3, 0.4, 0.3]),
        "CREATED_BY": "SYSTEM",
        "CREATED_TS": po_dates.astype(str),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_PO_HEADER.csv"
    })
    write_csv_split(erp_po_header, "RAW_ERP_PO_HEADER")

    po_line_rows = []
    pl_id_seq = 1
    for po_id in po_ids:
        n_lines = random.randint(1, 5)
        for ln in range(1, n_lines + 1):
            po_line_rows.append({
                "PO_LINE_ID": f"PL{pl_id_seq:08d}",
                "PO_ID": po_id,
                "LINE_NO": str(ln),
                "MATERIAL_ID": random.choice(material_ids),
                "ORDER_QTY": str(random.randint(100, 10000)),
                "ORDER_UOM": random.choice(["KG", "PCS"]),
                "UNIT_PRICE": str(round(random.uniform(100, 2000), 2)),
                "LINE_AMOUNT": "",  # can be calculated in STG/INT
                "PROMISED_DELIVERY_DATE": np.random.choice(DATE_RANGE).astype(str),
                "EXPECTED_ARRIVAL_DATE": np.random.choice(DATE_RANGE).astype(str),
                "SEASONAL_PEAK_FLAG": random.choice(["Y", "N"]),
                "LINE_STATUS": np.random.choice(["OPEN", "RECEIVED", "PARTIAL"], p=[0.4, 0.4, 0.2]),
                "DW_CREATED_TS": now_ts,
                "DW_LAST_UPDATED_TS": now_ts,
                "DW_BATCH_ID": batch_id,
                "DW_SOURCE_FILE_NAME": "ERP_PO_LINE.csv"
            })
            pl_id_seq += 1
    erp_po_line = pd.DataFrame(po_line_rows)
    write_csv_split(erp_po_line, "RAW_ERP_PO_LINE")

    # 2.3 GRN
    po_line_ids = erp_po_line["PO_LINE_ID"].values
    erp_grn = pd.DataFrame({
        "GRN_ID": [f"GRN{i:08d}" for i in range(1, N_GRN + 1)],
        "GRN_NO": [f"GR{random.randint(100000,999999)}" for _ in range(N_GRN)],
        "GRN_DATE": np.random.choice(DATE_RANGE, N_GRN).astype(str),
        "PO_LINE_ID": np.random.choice(po_line_ids, N_GRN),
        "VENDOR_ID": np.random.choice(vendor_ids, N_GRN),
        "MATERIAL_ID": np.random.choice(material_ids, N_GRN),
        "PLANT_ID": np.random.choice(plant_ids, N_GRN),
        "RECEIVED_QTY": np.random.randint(50, 10000, N_GRN).astype(str),
        "REJECTED_QTY": np.random.randint(0, 200, N_GRN).astype(str),
        "ACCEPTED_QTY": np.random.randint(50, 10000, N_GRN).astype(str),
        "EXPECTED_ARRIVAL_DATE": np.random.choice(DATE_RANGE, N_GRN).astype(str),
        "ACTUAL_ARRIVAL_DATE": np.random.choice(DATE_RANGE, N_GRN).astype(str),
        "DELIVERY_DELAY_DAYS": np.random.randint(-5, 10, N_GRN).astype(str),
        "LEAD_TIME_DAYS": np.random.randint(5, 40, N_GRN).astype(str),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_GRN.csv"
    })
    write_csv_split(erp_grn, "RAW_ERP_GRN")

    # --------------------------------------------------------
    # 3. PRODUCTION & INVENTORY
    # --------------------------------------------------------

    # 3.1 VEHICLE PRODUCTION
    erp_vehicle_production = pd.DataFrame({
        "VEHICLE_ID": [f"VH{i:07d}" for i in range(1, N_VEHICLE_PRODUCTION + 1)],
        "VIN": [f"VIN{random_string(12)}" for _ in range(N_VEHICLE_PRODUCTION)],
        "VARIANT_ID": np.random.choice(variant_ids, N_VEHICLE_PRODUCTION),
        "PLANT_ID": np.random.choice(plant_ids, N_VEHICLE_PRODUCTION),
        "PRODUCTION_DATE": np.random.choice(DATE_RANGE, N_VEHICLE_PRODUCTION).astype(str),
        "PRODUCTION_BATCH_NO": [f"PB{random.randint(1,9999):04d}" for _ in range(N_VEHICLE_PRODUCTION)],
        "ENGINE_NO": [f"EN{random.randint(1,9999999):07d}" for _ in range(N_VEHICLE_PRODUCTION)],
        "CHASSIS_NO": [f"CH{random.randint(1,9999999):07d}" for _ in range(N_VEHICLE_PRODUCTION)],
        "WARRANTY_START_DATE": np.random.choice(DATE_RANGE, N_VEHICLE_PRODUCTION).astype(str),
        "WARRANTY_DURATION_MONTHS": np.random.randint(24, 60, N_VEHICLE_PRODUCTION).astype(str),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_VEHICLE_PRODUCTION.csv"
    })
    write_csv_split(erp_vehicle_production, "RAW_ERP_VEHICLE_PRODUCTION")

    # 3.2 PRODUCTION DAILY
    erp_production_daily = pd.DataFrame({
        "PROD_RECORD_ID": [f"PR{i:07d}" for i in range(1, N_PRODUCTION_DAILY + 1)],
        "PROD_DATE": np.random.choice(DATE_RANGE, N_PRODUCTION_DAILY).astype(str),
        "PLANT_ID": np.random.choice(plant_ids, N_PRODUCTION_DAILY),
        "VARIANT_ID": np.random.choice(variant_ids, N_PRODUCTION_DAILY),
        "SHIFT_CODE": np.random.choice(["A", "B", "C"], N_PRODUCTION_DAILY),
        "UNITS_PRODUCED": np.random.randint(10, 80, N_PRODUCTION_DAILY).astype(str),
        "UNITS_REJECTED": np.random.randint(0, 5, N_PRODUCTION_DAILY).astype(str),
        "UNITS_REWORKED": np.random.randint(0, 3, N_PRODUCTION_DAILY).astype(str),
        "TOTAL_PRODUCTION_HRS": np.random.uniform(5, 24, N_PRODUCTION_DAILY).round(2).astype(str),
        "OEE_PERCENT": np.random.uniform(60, 95, N_PRODUCTION_DAILY).round(2).astype(str),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_PRODUCTION_DAILY.csv"
    })
    write_csv_split(erp_production_daily, "RAW_ERP_PRODUCTION_DAILY")

    # 3.3 PLANT INVENTORY SNAP
    erp_plant_inventory = pd.DataFrame({
        "SNAPSHOT_ID": [f"PI{i:07d}" for i in range(1, N_PLANT_INV + 1)],
        "SNAPSHOT_DATE": np.random.choice(DATE_RANGE, N_PLANT_INV).astype(str),
        "PLANT_ID": np.random.choice(plant_ids, N_PLANT_INV),
        "VARIANT_ID": np.random.choice(variant_ids, N_PLANT_INV),
        "OPENING_STOCK_UNITS": np.random.randint(0, 200, N_PLANT_INV).astype(str),
        "UNITS_PRODUCED": np.random.randint(0, 100, N_PLANT_INV).astype(str),
        "UNITS_DISPATCHED": np.random.randint(0, 100, N_PLANT_INV).astype(str),
        "CLOSING_STOCK_UNITS": np.random.randint(0, 200, N_PLANT_INV).astype(str),
        "AGEING_30_PLUS_UNITS": np.random.randint(0, 50, N_PLANT_INV).astype(str),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_PLANT_INVENTORY_SNAP.csv"
    })
    write_csv_split(erp_plant_inventory, "RAW_ERP_PLANT_INVENTORY_SNAP")

    # 3.4 VEHICLE DISPATCH
    erp_vehicle_dispatch = pd.DataFrame({
        "DISPATCH_ID": [f"DP{i:07d}" for i in range(1, N_DISPATCH + 1)],
        "DISPATCH_DOC_NO": [f"DD{random.randint(100000,999999)}" for _ in range(N_DISPATCH)],
        "DISPATCH_DATE": np.random.choice(DATE_RANGE, N_DISPATCH).astype(str),
        "PLANT_ID": np.random.choice(plant_ids, N_DISPATCH),
        "DEALER_ID": np.random.choice(dealer_ids, N_DISPATCH),
        "VARIANT_ID": np.random.choice(variant_ids, N_DISPATCH),
        "DISPATCH_QTY": np.random.randint(1, 20, N_DISPATCH).astype(str),
        "FREIGHT_COST": np.random.uniform(5000, 50000, N_DISPATCH).round(2).astype(str),
        "EXPECTED_ARRIVAL_DATE": np.random.choice(DATE_RANGE, N_DISPATCH).astype(str),
        "ACTUAL_ARRIVAL_DATE": np.random.choice(DATE_RANGE, N_DISPATCH).astype(str),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_VEHICLE_DISPATCH.csv"
    })
    write_csv_split(erp_vehicle_dispatch, "RAW_ERP_VEHICLE_DISPATCH")

    # --------------------------------------------------------
    # 4. SALES & DEALER OPERATIONS
    # --------------------------------------------------------

    # 4.1 DEALER STOCK SNAP
    erp_dealer_stock = pd.DataFrame({
        "SNAPSHOT_ID": [f"DS{i:07d}" for i in range(1, N_DEALER_STOCK + 1)],
        "SNAPSHOT_DATE": np.random.choice(DATE_RANGE, N_DEALER_STOCK).astype(str),
        "DEALER_ID": np.random.choice(dealer_ids, N_DEALER_STOCK),
        "VARIANT_ID": np.random.choice(variant_ids, N_DEALER_STOCK),
        "OPENING_STOCK_UNITS": np.random.randint(0, 80, N_DEALER_STOCK).astype(str),
        "UNITS_RECEIVED": np.random.randint(0, 50, N_DEALER_STOCK).astype(str),
        "UNITS_SOLD": np.random.randint(0, 50, N_DEALER_STOCK).astype(str),
        "UNITS_TRANSFERRED_OUT": np.random.randint(0, 10, N_DEALER_STOCK).astype(str),
        "CLOSING_STOCK_UNITS": np.random.randint(0, 80, N_DEALER_STOCK).astype(str),
        "TEST_DRIVE_VEHICLES": np.random.randint(0, 10, N_DEALER_STOCK).astype(str),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_DEALER_STOCK_SNAP.csv"
    })
    write_csv_split(erp_dealer_stock, "RAW_ERP_DEALER_STOCK_SNAP")

    # 4.2 BOOKING
    erp_booking = pd.DataFrame({
        "BOOKING_ID": [f"BK{i:08d}" for i in range(1, N_BOOKING + 1)],
        "BOOKING_NO": [f"BKNO{i:08d}" for i in range(1, N_BOOKING + 1)],
        "BOOKING_DATE": np.random.choice(DATE_RANGE, N_BOOKING).astype(str),
        "DEALER_ID": np.random.choice(dealer_ids, N_BOOKING),
        "SHOWROOM_ID": np.random.choice(erp_showroom["SHOWROOM_ID"], N_BOOKING),
        "VARIANT_ID": np.random.choice(variant_ids, N_BOOKING),
        "CUSTOMER_ID": np.random.choice(cust_ids, N_BOOKING),
        "SALESPERSON_ID": np.random.choice(erp_salesperson["SALESPERSON_ID"], N_BOOKING),
        "BOOKING_AMOUNT": np.random.uniform(50000, 200000, N_BOOKING).round(2).astype(str),
        "BOOKING_STATUS": np.random.choice(["OPEN", "INVOICED", "CANCELLED"], N_BOOKING, p=[0.3, 0.6, 0.1]),
        "EXPECTED_DELIVERY_DATE": np.random.choice(DATE_RANGE, N_BOOKING).astype(str),
        "CANCEL_DATE": "",
        "CANCEL_REASON": "",
        "PREDICTED_CONVERSION_PROB": np.random.uniform(0.5, 0.99, N_BOOKING).round(3).astype(str),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_BOOKING.csv"
    })
    write_csv_split(erp_booking, "RAW_ERP_BOOKING")

    # 4.3 SALES INVOICE HDR
    inv_ids = [f"INV{i:08d}" for i in range(1, N_INVOICE + 1)]
    inv_dates = np.random.choice(DATE_RANGE, N_INVOICE)
    erp_sales_invoice_hdr = pd.DataFrame({
        "INVOICE_ID": inv_ids,
        "INVOICE_NO": [f"SI{random.randint(100000,999999)}" for _ in range(N_INVOICE)],
        "INVOICE_DATE": inv_dates.astype(str),
        "DELIVERY_DATE": (inv_dates + pd.to_timedelta(np.random.randint(0, 10, N_INVOICE), unit='D')).astype(str),
        "DEALER_ID": np.random.choice(dealer_ids, N_INVOICE),
        "CUSTOMER_ID": np.random.choice(cust_ids, N_INVOICE),
        "SALESPERSON_ID": np.random.choice(erp_salesperson["SALESPERSON_ID"], N_INVOICE),
        "PAYMENT_MODE": np.random.choice(["Cash", "Loan", "Lease"], N_INVOICE),
        "PAYMENT_STATUS": np.random.choice(["PAID", "PARTIAL", "PENDING"], N_INVOICE, p=[0.7, 0.2, 0.1]),
        "DOWN_PAYMENT_AMOUNT": np.random.uniform(50000, 300000, N_INVOICE).round(2).astype(str),
        "LOAN_AMOUNT": np.random.uniform(200000, 1000000, N_INVOICE).round(2).astype(str),
        "TOTAL_EXSHOWROOM_PRICE": np.random.uniform(300000, 1500000, N_INVOICE).round(2).astype(str),
        "TOTAL_DISCOUNT_AMOUNT": np.random.uniform(0, 100000, N_INVOICE).round(2).astype(str),
        "TOTAL_ONROAD_PRICE": np.random.uniform(400000, 1700000, N_INVOICE).round(2).astype(str),
        "BOOKING_ID": np.random.choice(erp_booking["BOOKING_ID"], N_INVOICE),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_SALES_INVOICE_HDR.csv"
    })
    write_csv_split(erp_sales_invoice_hdr, "RAW_ERP_SALES_INVOICE_HDR")

    # 4.4 SALES INVOICE LINE (1 per invoice)
    erp_sales_invoice_line = pd.DataFrame({
        "INVOICE_LINE_ID": [f"INVL{i:08d}" for i in range(1, N_INVOICE + 1)],
        "INVOICE_ID": inv_ids,
        "VEHICLE_ID": np.random.choice(erp_vehicle_production["VEHICLE_ID"], N_INVOICE),
        "VARIANT_ID": np.random.choice(variant_ids, N_INVOICE),
        "EXSHOWROOM_PRICE": erp_sales_invoice_hdr["TOTAL_EXSHOWROOM_PRICE"],
        "DISCOUNT_AMOUNT": erp_sales_invoice_hdr["TOTAL_DISCOUNT_AMOUNT"],
        "ONROAD_PRICE": erp_sales_invoice_hdr["TOTAL_ONROAD_PRICE"],
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_SALES_INVOICE_LINE.csv"
    })
    write_csv_split(erp_sales_invoice_line, "RAW_ERP_SALES_INVOICE_LINE")

    # --------------------------------------------------------
    # 5. SERVICE / WARRANTY / FEEDBACK
    # --------------------------------------------------------

    # 5.1 FAULT CODE & SERVICE JOB TYPE
    N_FAULT_CODES = 60
    fault_code_ids = [f"FC{i:04d}" for i in range(1, N_FAULT_CODES + 1)]
    erp_fault_code = pd.DataFrame({
        "FAULT_CODE_ID": fault_code_ids,
        "FAULT_CODE": [f"F{i:04d}" for i in range(1, N_FAULT_CODES + 1)],
        "FAULT_DESCRIPTION": [f"Fault description {i}" for i in range(1, N_FAULT_CODES + 1)],
        "SEVERITY_LEVEL": np.random.choice(["LOW", "MEDIUM", "HIGH"], N_FAULT_CODES),
        "SYSTEM_AREA": np.random.choice(["Engine", "Transmission", "Electrical", "Body"], N_FAULT_CODES),
        "IS_MANUFACTURING_DEFECT_FLAG": np.random.choice(["Y", "N"], N_FAULT_CODES, p=[0.2, 0.8]),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_FAULT_CODE.csv"
    })
    write_csv_split(erp_fault_code, "RAW_ERP_FAULT_CODE")

    N_JOB_TYPES = 40
    job_type_ids = [f"JT{i:04d}" for i in range(1, N_JOB_TYPES + 1)]
    erp_service_job_type = pd.DataFrame({
        "SERVICE_JOB_TYPE_ID": job_type_ids,
        "JOB_TYPE_CODE": [f"JT{i:04d}" for i in range(1, N_JOB_TYPES + 1)],
        "JOB_TYPE_NAME": [f"Job Type {i}" for i in range(1, N_JOB_TYPES + 1)],
        "JOB_CATEGORY": np.random.choice(["Periodic", "Repair", "Bodyshop", "Electrical"], N_JOB_TYPES),
        "IS_WARRANTY_ELIGIBLE_FLAG": np.random.choice(["Y", "N"], N_JOB_TYPES, p=[0.5, 0.5]),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_SERVICE_JOB_TYPE.csv"
    })
    write_csv_split(erp_service_job_type, "RAW_ERP_SERVICE_JOB_TYPE")

    # 5.2 SERVICE ORDER HDR
    svc_dates = np.random.choice(DATE_RANGE, N_SERVICE_ORDERS)
    erp_service_order_hdr = pd.DataFrame({
        "SERVICE_ORDER_ID": [f"SO{i:08d}" for i in range(1, N_SERVICE_ORDERS + 1)],
        "SERVICE_ORDER_NO": [f"SO{random.randint(100000,999999)}" for _ in range(N_SERVICE_ORDERS)],
        "SERVICE_DATE": svc_dates.astype(str),
        "SERVICE_CENTER_ID": np.random.choice(erp_service_center["SERVICE_CENTER_ID"], N_SERVICE_ORDERS),
        "DEALER_ID": np.random.choice(dealer_ids, N_SERVICE_ORDERS),
        "VEHICLE_ID": np.random.choice(erp_vehicle_production["VEHICLE_ID"], N_SERVICE_ORDERS),
        "CUSTOMER_ID": np.random.choice(cust_ids, N_SERVICE_ORDERS),
        "KM_ODOMETER_READING": np.random.randint(5000, 200000, N_SERVICE_ORDERS).astype(str),
        "LABOUR_AMOUNT": np.random.uniform(2000, 20000, N_SERVICE_ORDERS).round(2).astype(str),
        "PARTS_AMOUNT": np.random.uniform(1000, 50000, N_SERVICE_ORDERS).round(2).astype(str),
        "TOTAL_BILL_AMOUNT": np.random.uniform(3000, 65000, N_SERVICE_ORDERS).round(2).astype(str),
        "IS_WARRANTY_COVERED_FLAG": np.random.choice(["Y", "N"], N_SERVICE_ORDERS, p=[0.3, 0.7]),
        "CUSTOMER_SATISFACTION_SCORE": np.random.uniform(6, 10, N_SERVICE_ORDERS).round(1).astype(str),
        "NPS_SCORE": np.random.randint(0, 10, N_SERVICE_ORDERS).astype(str),
        "NEXT_SERVICE_DUE_DATE": (svc_dates + pd.to_timedelta(np.random.randint(90, 365, N_SERVICE_ORDERS), unit='D')).astype(str),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_SERVICE_ORDER_HDR.csv"
    })
    write_csv_split(erp_service_order_hdr, "RAW_ERP_SERVICE_ORDER_HDR")

    # 5.3 SERVICE ORDER LINE
    erp_service_order_line = pd.DataFrame({
        "SERVICE_ORDER_LINE_ID": [f"SOL{i:08d}" for i in range(1, N_SERVICE_LINES + 1)],
        "SERVICE_ORDER_ID": np.random.choice(erp_service_order_hdr["SERVICE_ORDER_ID"], N_SERVICE_LINES),
        "LINE_NO": np.random.randint(1, 5, N_SERVICE_LINES).astype(str),
        "FAULT_CODE_ID": np.random.choice(fault_code_ids, N_SERVICE_LINES),
        "REPLACED_MATERIAL_ID": np.random.choice(material_ids, N_SERVICE_LINES),
        "IS_MANUFACTURING_DEFECT_FLAG": np.random.choice(["Y", "N"], N_SERVICE_LINES, p=[0.2, 0.8]),
        "WARRANTY_FLAG": np.random.choice(["Y", "N"], N_SERVICE_LINES, p=[0.4, 0.6]),
        "REPAIR_HOURS": np.random.uniform(0.5, 8, N_SERVICE_LINES).round(1).astype(str),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_SERVICE_ORDER_LINE.csv"
    })
    write_csv_split(erp_service_order_line, "RAW_ERP_SERVICE_ORDER_LINE")

    # 5.4 WARRANTY CLAIM
    erp_warranty_claim = pd.DataFrame({
        "CLAIM_ID": [f"WC{i:08d}" for i in range(1, N_WARRANTY + 1)],
        "CLAIM_NO": [f"WC{random.randint(100000,999999)}" for _ in range(N_WARRANTY)],
        "CLAIM_DATE": np.random.choice(DATE_RANGE, N_WARRANTY).astype(str),
        "SERVICE_ORDER_ID": np.random.choice(erp_service_order_hdr["SERVICE_ORDER_ID"], N_WARRANTY),
        "VEHICLE_ID": np.random.choice(erp_vehicle_production["VEHICLE_ID"], N_WARRANTY),
        "DEALER_ID": np.random.choice(dealer_ids, N_WARRANTY),
        "FAULT_CODE_ID": np.random.choice(fault_code_ids, N_WARRANTY),
        "CLAIM_AMOUNT": np.random.uniform(1000, 50000, N_WARRANTY).round(2).astype(str),
        "APPROVED_AMOUNT": np.random.uniform(500, 40000, N_WARRANTY).round(2).astype(str),
        "CLAIM_STATUS": np.random.choice(["OPEN", "APPROVED", "REJECTED"], N_WARRANTY),
        "REJECTION_REASON": "",
        "MANUFACTURING_DEFECT_FLAG": np.random.choice(["Y", "N"], N_WARRANTY, p=[0.3, 0.7]),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_WARRANTY_CLAIM.csv"
    })
    write_csv_split(erp_warranty_claim, "RAW_ERP_WARRANTY_CLAIM")

    # 5.5 CUSTOMER FEEDBACK
    erp_customer_feedback = pd.DataFrame({
        "FEEDBACK_ID": [f"FB{i:08d}" for i in range(1, N_FEEDBACK + 1)],
        "FEEDBACK_DATE": np.random.choice(DATE_RANGE, N_FEEDBACK).astype(str),
        "CUSTOMER_ID": np.random.choice(cust_ids, N_FEEDBACK),
        "VEHICLE_ID": np.random.choice(erp_vehicle_production["VEHICLE_ID"], N_FEEDBACK),
        "DEALER_ID": np.random.choice(dealer_ids, N_FEEDBACK),
        "SERVICE_ORDER_ID": np.random.choice(erp_service_order_hdr["SERVICE_ORDER_ID"], N_FEEDBACK),
        "OVERALL_SATISFACTION_SCORE": np.random.uniform(1, 10, N_FEEDBACK).round(1).astype(str),
        "RECOMMEND_SCORE": np.random.randint(0, 10, N_FEEDBACK).astype(str),
        "FEATURE_REQUEST_CATEGORY": np.random.choice(["Comfort", "Safety", "Infotainment", "Performance", "Design"], N_FEEDBACK),
        "FEATURE_REQUEST_TEXT": "Sample feedback text",
        "COMPLAINT_FLAG": np.random.choice(["Y", "N"], N_FEEDBACK, p=[0.3, 0.7]),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_CUSTOMER_FEEDBACK.csv"
    })
    write_csv_split(erp_customer_feedback, "RAW_ERP_CUSTOMER_FEEDBACK")

    # --------------------------------------------------------
    # 6. FORECAST & BACKLOG
    # --------------------------------------------------------

    erp_demand_forecast = pd.DataFrame({
        "FORECAST_RUN_ID": [f"FRC{random.randint(1,50):03d}" for _ in range(N_FORECAST)],
        "FORECAST_SOURCE": np.random.choice(["StatModel", "MLModel", "Manual"], N_FORECAST),
        "FORECAST_DATE": np.random.choice(DATE_RANGE, N_FORECAST).astype(str),
        "DEALER_ID": np.random.choice(dealer_ids, N_FORECAST),
        "VARIANT_ID": np.random.choice(variant_ids, N_FORECAST),
        "FORECAST_DEMAND_UNITS": np.random.randint(1, 100, N_FORECAST).astype(str),
        "FORECAST_GENERATED_DATE": np.random.choice(DATE_RANGE, N_FORECAST).astype(str),
        "SEASONAL_PEAK_FLAG": np.random.choice(["Y", "N"], N_FORECAST, p=[0.2, 0.8]),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_DEMAND_FORECAST.csv"
    })
    write_csv_split(erp_demand_forecast, "RAW_ERP_DEMAND_FORECAST")

    erp_backlog_snapshot = pd.DataFrame({
        "BACKLOG_RUN_ID": [f"BLR{random.randint(1,50):03d}" for _ in range(N_BACKLOG)],
        "AS_OF_DATE": np.random.choice(DATE_RANGE, N_BACKLOG).astype(str),
        "PLANT_ID": np.random.choice(plant_ids, N_BACKLOG),
        "VARIANT_ID": np.random.choice(variant_ids, N_BACKLOG),
        "TOTAL_OPEN_BOOKINGS_UNITS": np.random.randint(0, 500, N_BACKLOG).astype(str),
        "NEAR_TERM_FORECAST_30D_UNITS": np.random.randint(0, 500, N_BACKLOG).astype(str),
        "PLANT_FINISHED_STOCK_UNITS": np.random.randint(0, 500, N_BACKLOG).astype(str),
        "DEALER_STOCK_UNITS": np.random.randint(0, 500, N_BACKLOG).astype(str),
        "IN_TRANSIT_VEHICLE_UNITS": np.random.randint(0, 500, N_BACKLOG).astype(str),
        "AVAILABLE_VEHICLE_SUPPLY_UNITS": np.random.randint(0, 500, N_BACKLOG).astype(str),
        "PLANNED_PRODUCTION_30D_UNITS": np.random.randint(0, 500, N_BACKLOG).astype(str),
        "EXPECTED_VENDOR_SUPPLY_30D_UOM": np.random.randint(0, 500, N_BACKLOG).astype(str),
        "BACKLOG_UNITS": np.random.randint(0, 500, N_BACKLOG).astype(str),
        "EXPECTED_AVG_DELAY_DAYS": np.random.randint(0, 60, N_BACKLOG).astype(str),
        "RECOMMENDED_EXTRA_PRODUCTION_UNITS": np.random.randint(0, 500, N_BACKLOG).astype(str),
        "RECOMMENDED_EXTRA_PROCUREMENT_UOM": np.random.randint(0, 500, N_BACKLOG).astype(str),
        "SEASONAL_PEAK_FLAG": np.random.choice(["Y", "N"], N_BACKLOG, p=[0.2, 0.8]),
        "DW_CREATED_TS": now_ts,
        "DW_LAST_UPDATED_TS": now_ts,
        "DW_BATCH_ID": batch_id,
        "DW_SOURCE_FILE_NAME": "ERP_BACKLOG_SNAPSHOT.csv"
    })
    write_csv_split(erp_backlog_snapshot, "RAW_ERP_BACKLOG_SNAPSHOT")

    print("\n✅ RAW CSV generation complete. Files are in:", os.path.abspath(BASE_PATH))


if __name__ == "__main__":
    main()
