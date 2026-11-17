import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import math
from datetime import datetime, date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_js_eval import get_geolocation

# --- PAGE SETUP ---
st.set_page_config(
    page_title="JanVaani | Smart Civic Complaint Portal",
    page_icon="🏙️",
    layout="wide"
)

# --- DATABASE SETUP (SQLite - Permanent & Scalable) ---
DB_FILE = "civic_complaints.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS complaints (
            id TEXT PRIMARY KEY,
            citizen_name TEXT,
            mobile TEXT,
            email TEXT,
            anonymous INTEGER,
            language TEXT,
            title TEXT,
            category TEXT,
            description TEXT,
            date_noticed TEXT,
            duration TEXT,
            people_affected TEXT,
            state TEXT,
            district TEXT,
            city TEXT,
            area TEXT,
            landmark TEXT,
            pincode TEXT,
            latitude REAL,
            longitude REAL,
            severity INTEGER,
            safety_risk TEXT,
            vulnerable_affected TEXT,
            recurring TEXT,
            getting_worse TEXT,
            reported_before TEXT,
            prev_id TEXT,
            preferred_resolution TEXT,
            frequency INTEGER DEFAULT 1,
            status TEXT DEFAULT 'Open',
            timestamp TEXT
        )
    ''')
    # Seed initial realistic sample data (Mysuru) if database is empty
    c.execute("SELECT COUNT(*) FROM complaints")
    if c.fetchone()[0] == 0:
        samples = [
            ("CMP-1001", "Ramesh Kumar", "9845012345", "ramesh@example.com", 0, "Kannada",
             "Major water pipeline burst near KR Hospital", "Water Supply & Leakage",
             "Main drinking water pipeline burst flooding the entire road near KR Hospital gate.",
             "2026-11-16", "1–3 days", "More than 100 people", "Karnataka", "Mysuru", "Mysuru",
             "Sayyaji Rao Road", "Opposite KR Hospital Main Gate", "570001", 12.3131, 76.6502,
             5, "Yes", "Yes", "Yes", "Yes", "No", "", "Immediate pipe repair", 3, "Open", "2026-11-16 09:30"),
            ("CMP-1002", "Anonymous Citizen", "9900112233", "", 1, "English",
             "Deep crater pothole at Metropole Circle", "Roads & Potholes",
             "Large deep pothole right at the junction causing two-wheeler skids.",
             "2026-11-15", "4–7 days", "50–100 people", "Karnataka", "Mysuru", "Mysuru",
             "Vontikoppal", "Near Metropole Circle", "570005", 12.3105, 76.6412,
             4, "Yes", "Unknown", "Yes", "Yes", "No", "", "Fill and asphalt the road", 2, "Open", "2026-11-16 11:15"),
            ("CMP-1003", "Priya Nair", "9880054321", "priya@example.com", 0, "English",
             "Overflowing garbage bin near Devaraja Market", "Garbage & Sanitation",
             "Garbage has not been cleared for 4 days; foul smell and stray dogs gathering.",
             "2026-11-14", "4–7 days", "50–100 people", "Karnataka", "Mysuru", "Mysuru",
             "Devaraja Mohalla", "Near Devaraja Market North Gate", "570001", 12.3122, 76.6515,
             3, "No", "Yes", "Yes", "No", "No", "", "Daily garbage clearance", 1, "Open", "2026-11-17 08:15")
        ]
        c.executemany('''
            INSERT INTO complaints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', samples)
    conn.commit()
    conn.close()

def load_complaints():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM complaints", conn)
    conn.close()
    return df

def insert_complaint(data_tuple):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO complaints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', data_tuple)
    conn.commit()
    conn.close()

def increment_frequency(complaint_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE complaints SET frequency = frequency + 1 WHERE id = ?", (complaint_id,))
    conn.commit()
    conn.close()

def update_status(complaint_id, new_status):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE complaints SET status = ? WHERE id = ?", (new_status, complaint_id))
    conn.commit()
    conn.close()

init_db()

# --- SMART ALGORITHMS (DUPLICATE, HOTSPOT, PRIORITIZATION) ---

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def check_duplicate(new_cat, new_title, new_desc, new_lat, new_lon, new_area, df):
    same_cat = df[(df["category"] == new_cat) & (df["status"] != "Resolved")]
    if same_cat.empty:
        return None, 0.0

    candidates = []
    for _, row in same_cat.iterrows():
        # Match by GPS proximity (<= 300m) OR exact same Area/Locality if GPS wasn't used
        gps_match = False
        if new_lat != 0.0 and row["latitude"] != 0.0:
            dist = haversine_distance(new_lat, new_lon, row["latitude"], row["longitude"])
            if dist <= 300:
                gps_match = True
        area_match = (new_area.strip().lower() == str(row["area"]).strip().lower())

        if gps_match or area_match:
            candidates.append((row["id"], f"{row['title']} {row['description']}"))

    if not candidates:
        return None, 0.0

    corpus = [item[1] for item in candidates] + [f"{new_title} {new_desc}"]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf[-1], tfidf[:-1])[0]

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if best_score >= 0.40:
        return candidates[best_idx][0], best_score
    return None, best_score

def calculate_priority_score(row):
    # 1. Severity (1-5 scale -> max 35 pts)
    sev_pts = (row["severity"] / 5.0) * 35

    # 2. Frequency / Duplicate Boost (max 25 pts)
    freq_pts = min(row["frequency"] / 5.0, 1.0) * 25

    # 3. Public Safety Risk & Escalation (max 20 pts)
    safety_pts = 12 if row["safety_risk"] == "Yes" else 0
    worse_pts = 5 if row["getting_worse"] == "Yes" else 0
    vuln_pts = 3 if row["vulnerable_affected"] == "Yes" else 0

    # 4. Public Impact / People Affected (max 20 pts)
    impact_map = {
        "Less than 10 people": 5,
        "10–50 people": 10,
        "50–100 people": 15,
        "More than 100 people": 20
    }
    impact_pts = impact_map.get(row["people_affected"], 8)

    return round(sev_pts + freq_pts + safety_pts + worse_pts + vuln_pts + impact_pts, 1)

def detect_hotspots(df, radius_m=500, min_cluster=2):
    flags = []
    valid_df = df[df["status"] != "Resolved"]
    for _, r1 in df.iterrows():
        if r1["latitude"] == 0.0 or r1["status"] == "Resolved":
            flags.append("Normal")
            continue
        count = 0
        for _, r2 in valid_df.iterrows():
            if r2["latitude"] != 0.0:
                if haversine_distance(r1["latitude"], r1["longitude"], r2["latitude"], r2["longitude"]) <= radius_m:
                    count += 1
        flags.append("🔥 HOTSPOT" if count >= min_cluster else "Normal")
    return flags

# --- SESSION STATE FOR GPS COORDINATES ---
if "gps_lat" not in st.session_state:
    st.session_state.gps_lat = 0.0
if "gps_lon" not in st.session_state:
    st.session_state.gps_lon = 0.0
if "fetch_gps" not in st.session_state:
    st.session_state.fetch_gps = False

# --- HEADER & NAVIGATION ---
st.title("🏙️ Smart Civic Complaint & Hotspot Portal")
st.caption("Simple, transparent public grievance reporting with automated duplicate merging, hotspot alerts & priority ranking.")

tab_citizen, tab_track, tab_admin = st.tabs([
    "📝 Report an Issue",
    "🔍 Track My Complaint",
    "📊 Municipal Admin Dashboard"
])

# ==============================================================================
# TAB 1: CITIZEN COMPLAINT FORM (Standard, Concise & User-Friendly)
# ==============================================================================
with tab_citizen:
    st.markdown("### Step 1: Location Details *(Section C)*")
    st.write("Choose **either** automatic GPS location **or** enter your address below:")

    loc_col1, loc_col2 = st.columns([1, 2])
    with loc_col1:
        if st.button("📍 Use My Current Location", use_container_width=True, type="secondary"):
            st.session_state.fetch_gps = True

    if st.session_state.fetch_gps:
        loc_data = get_geolocation()
        if loc_data and "coords" in loc_data:
            st.session_state.gps_lat = round(loc_data["coords"]["latitude"], 6)
            st.session_state.gps_lon = round(loc_data["coords"]["longitude"], 6)
            st.session_state.fetch_gps = False
            st.success(f"GPS Captured: {st.session_state.gps_lat}, {st.session_state.gps_lon}")
        else:
            st.info("Waiting for browser location permission... Click 'Allow' in your browser popup.")

    with loc_col2:
        if st.session_state.gps_lat != 0.0:
            st.info(f"✅ **GPS Coordinates Attached:** Lat `{st.session_state.gps_lat}`, Lon `{st.session_state.gps_lon}`")
        else:
            st.caption("Tip: If you are at the problem spot, click **Use My Current Location** for exact map pinning.")

    with st.form("citizen_complaint_form", clear_on_submit=False):
        # Manual Address Fields
        c1, c2, c3 = st.columns(3)
        state = c1.selectbox("State *", ["Karnataka", "Maharashtra", "Tamil Nadu", "Kerala", "Delhi", "Other"])
        district = c2.text_input("District *", value="Mysuru")
        city = c3.text_input("City / Town *", value="Mysuru")

        c4, c5, c6 = st.columns([2, 2, 1])
        area = c4.text_input("Area / Locality *", placeholder="e.g., Kuvempunagar, VV Mohalla")
        landmark = c5.text_input("Street / Nearest Landmark *", placeholder="e.g., Near Akshay Bhandar Signal")
        pincode = c6.text_input("PIN Code", placeholder="570023")

        # Hidden/Editable GPS fallback for testing or manual adjustment
        g1, g2 = st.columns(2)
        lat_input = g1.number_input("GPS Latitude (Auto-filled or optional)", value=float(st.session_state.gps_lat), format="%.6f")
        lon_input = g2.number_input("GPS Longitude (Auto-filled or optional)", value=float(st.session_state.gps_lon), format="%.6f")

        st.divider()

        # --- SECTION B: COMPLAINT DETAILS ---
        st.markdown("### Step 2: What is the Problem? *(Section B)*")
        b1, b2 = st.columns([2, 1])
        title = b1.text_input("Complaint Title *", placeholder="e.g., Streetlights not working on 4th Main Road")
        category = b2.selectbox("Complaint Category *", [
            "Roads & Potholes",
            "Water Supply & Leakage",
            "Garbage & Sanitation",
            "Streetlights & Electricity",
            "Drainage & Sewage",
            "Public Safety & Encroachment",
            "Other"
        ])

        description = st.text_area("Problem Description *", height=90, placeholder="Briefly describe what happened and how it is affecting people...")

        b3, b4, b5 = st.columns(3)
        duration = b3.selectbox("How long has this existed? *", ["Today", "1–3 days", "4–7 days", "More than a week", "Over a month"])
        people_affected = b4.selectbox("Estimated People Affected", ["Less than 10 people", "10–50 people", "50–100 people", "More than 100 people"])
        photo = b5.file_uploader("Upload Photo / Video (Recommended)", type=["jpg", "jpeg", "png", "mp4"])

        st.divider()

        # --- SECTION D: SEVERITY & IMPACT ---
        st.markdown("### Step 3: Urgency & Safety Impact *(Section D)*")
        severity_options = {
            1: "1 – Low (Minor inconvenience, little public impact)",
            2: "2 – Mild (Small issue that requires attention)",
            3: "3 – Moderate (Noticeable inconvenience to multiple people)",
            4: "4 – High (Major disruption or potential safety risk)",
            5: "5 – Critical (Immediate danger to public safety)"
        }
        severity_label = st.select_slider(
            "Severity Level *",
            options=list(severity_options.values()),
            value=severity_options[3]
        )
        severity_val = int(severity_label.split(" ")[0])

        d1, d2, d3 = st.columns(3)
        safety_risk = d1.radio("Risk to Public Safety? *", ["No", "Yes"], horizontal=True)
        recurring = d2.radio("Is this a recurring issue? *", ["No", "Yes", "Unknown"], horizontal=True)
        getting_worse = d3.radio("Is the issue getting worse?", ["No", "Yes", "Unknown"], horizontal=True)

        st.divider()

        # --- SECTION A: CITIZEN INFORMATION ---
        st.markdown("### Step 4: Your Contact Details *(Section A)*")
        anonymous = st.checkbox("Report Anonymously (Your name won't be shown publicly)")
        a1, a2 = st.columns(2)
        full_name = a1.text_input("Full Name *", placeholder="Enter your full name")
        mobile = a2.text_input("Mobile Number * (For SMS status updates)", placeholder="10-digit mobile number", max_chars=10)

        # --- OPTIONAL FIELDS (Sections A, B, D, E grouped cleanly) ---
        with st.expander("➕ Optional Additional Details (Language, Email, Date Noticed, Previous ID)"):
            opt1, opt2, opt3 = st.columns(3)
            email = opt1.text_input("Email Address (Optional)")
            language = opt2.selectbox("Preferred Language", ["English", "Kannada", "Hindi", "Tamil", "Telugu"])
            date_noticed = opt3.date_input("Date First Noticed", value=date.today())

            opt4, opt5 = st.columns(2)
            vulnerable_affected = opt4.selectbox("Affects Children / Elderly / Disabled?", ["Unknown", "Yes", "No"])
            reported_before = opt5.selectbox("Has this been reported before?", ["No", "Yes", "Unknown"])

            opt6, opt7 = st.columns(2)
            prev_id = opt6.text_input("Previous Complaint ID (if any)", placeholder="e.g., CMP-1001")
            preferred_resolution = opt7.text_input("Preferred Resolution", placeholder="e.g., Replace broken manhole cover")

        st.divider()

        # --- SECTION F: DECLARATION ---
        st.markdown("### Section F – Declaration")
        decl_1 = st.checkbox("I confirm that the information provided is accurate. *")
        decl_2 = st.checkbox("I agree to the use of the submitted location and complaint information for resolving the issue. *")

        submit_btn = st.form_submit_button("✅ Submit Complaint", type="primary", use_container_width=True)

        if submit_btn:
            # Validation
            if not area.strip() or not landmark.strip():
                st.error("Please enter the Area/Locality and Street/Landmark in Location Details.")
            elif not title.strip() or not description.strip():
                st.error("Please fill in the Complaint Title and Problem Description.")
            elif not anonymous and not full_name.strip():
                st.error("Please enter your Full Name or check 'Report Anonymously'.")
            elif len(mobile.strip()) < 10 or not mobile.strip().isdigit():
                st.error("Please enter a valid 10-digit Mobile Number.")
            elif not (decl_1 and decl_2):
                st.error("Please check both Declaration boxes in Section F before submitting.")
            else:
                df_existing = load_complaints()
                dup_id, sim_score = check_duplicate(category, title, description, lat_input, lon_input, area, df_existing)

                if dup_id:
                    increment_frequency(dup_id)
                    st.warning(
                        f"⚠️ **Similar Complaint Already Active ({int(sim_score*100)}% match)!**\n\n"
                        f"Your report has been linked to existing ticket **{dup_id}** in **{area}**. "
                        f"Its priority score has been automatically increased so authorities address it faster!"
                    )
                else:
                    new_id = f"CMP-{1001 + len(df_existing)}"
                    saved_name = "Anonymous Citizen" if anonymous else full_name.strip()
                    record = (
                        new_id, saved_name, mobile.strip(), email.strip(), int(anonymous), language,
                        title.strip(), category, description.strip(), str(date_noticed), duration,
                        people_affected, state, district.strip(), city.strip(), area.strip(),
                        landmark.strip(), pincode.strip(), float(lat_input), float(lon_input),
                        severity_val, safety_risk, vulnerable_affected, recurring, getting_worse,
                        reported_before, prev_id.strip(), preferred_resolution.strip(),
                        1, "Open", datetime.now().strftime("%Y-%m-%d %H:%M")
                    )
                    insert_complaint(record)
                    st.success(
                        f"🎉 **Complaint Submitted Successfully!**\n\n"
                        f"Your Complaint ID is **{new_id}**. Save this ID to track progress anytime."
                    )

# ==============================================================================
# TAB 2: CITIZEN COMPLAINT TRACKER
# ==============================================================================
with tab_track:
    st.subheader("🔍 Check Your Complaint Status")
    search_query = st.text_input("Enter your Complaint ID (e.g., CMP-1001) or Mobile Number:")
    if search_query.strip():
        df_all = load_complaints()
        q = search_query.strip().upper()
        matches = df_all[(df_all["id"].str.upper() == q) | (df_all["mobile"] == search_query.strip())]
        if matches.empty:
            st.warning("No matching complaint found. Please check your Complaint ID or Mobile Number.")
        else:
            for _, row in matches.iterrows():
                with st.container(border=True):
                    c_left, c_right = st.columns([3, 1])
                    c_left.markdown(f"#### {row['id']} — {row['title']}")
                    c_left.write(f"**Category:** {row['category']} | **Location:** {row['area']}, {row['landmark']} ({row['city']})")
                    c_left.write(f"**Description:** {row['description']}")
                    c_left.caption(f"Reported on {row['timestamp']} • Duplicate Reports Merged: {row['frequency']}")
                    c_right.metric("Current Status", row["status"])

# ==============================================================================
# TAB 3: MUNICIPAL ADMIN DASHBOARD
# ==============================================================================
with tab_admin:
    st.subheader("📊 Smart Prioritization & Hotspot Command Center")
    df = load_complaints()

    if df.empty:
        st.info("No complaints registered yet.")
    else:
        df["Priority_Score"] = df.apply(calculate_priority_score, axis=1)
        df["Hotspot_Status"] = detect_hotspots(df)
        df = df.sort_values(by=["status", "Priority_Score"], ascending=[False, False])

        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        open_df = df[df["status"] != "Resolved"]
        k1.metric("Active Issues", len(open_df))
        k2.metric("Total Citizen Reports", int(df["frequency"].sum()))
        k3.metric("High Priority (Score ≥ 70)", len(open_df[open_df["Priority_Score"] >= 70]))
        k4.metric("Active Hotspot Zones", len(open_df[open_df["Hotspot_Status"] == "🔥 HOTSPOT"]))

        st.divider()

        col_table, col_map = st.columns([1.5, 1])

        with col_table:
            st.markdown("#### 🚨 Prioritized Complaint Queue")
            show_cols = ["id", "Priority_Score", "Hotspot_Status", "status", "category", "area", "frequency", "safety_risk", "title"]
            st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

            # Quick Status Updater for Authorities
            st.markdown("#### 🛠️ Update Ticket Status")
            u1, u2, u3 = st.columns([2, 2, 1])
            sel_id = u1.selectbox("Select Ticket ID", df["id"].tolist())
            new_stat = u2.selectbox("New Status", ["Open", "In Progress", "Resolved"])
            if u3.button("Update", use_container_width=True):
                update_status(sel_id, new_stat)
                st.success(f"{sel_id} updated to {new_stat}")
                st.rerun()

        with col_map:
            st.markdown("#### 📍 Geographic Hotspot Map")
            map_df = df[df["latitude"] != 0.0][["latitude", "longitude"]].copy()
            if not map_df.empty:
                st.map(map_df, zoom=13)
            else:
                st.info("No GPS-tagged complaints to display on map yet.")