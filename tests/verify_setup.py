import sys

def check_import(library_name, import_name=None):
    if import_name is None:
        import_name = library_name
    try:
        __import__(import_name)
        print(f"  ✅  {library_name}")
        return True
    except ImportError:
        print(f"  ❌  {library_name}")
        return False

def main():
    print("\n" + "="*50)
    print("  FLOOD RISK PROJECT — SETUP VERIFICATION")
    print("="*50)

    version = sys.version_info
    print(f"\n🐍 Python Version: {version.major}.{version.minor}.{version.micro}")

    print("\n📦 Checking Libraries...")
    print("-"*50)

    libraries = [
        ("numpy",           "numpy"),
        ("pandas",          "pandas"),
        ("matplotlib",      "matplotlib"),
        ("seaborn",         "seaborn"),
        ("scikit-learn",    "sklearn"),
        ("xgboost",         "xgboost"),
        ("streamlit",       "streamlit"),
        ("folium",          "folium"),
        ("geopandas",       "geopandas"),
        ("shapely",         "shapely"),
        ("earthengine-api", "ee"),
        ("geemap",          "geemap"),
        ("plotly",          "plotly"),
        ("joblib",          "joblib"),
        ("Pillow",          "PIL"),
        ("python-dotenv",   "dotenv"),
        ("tqdm",            "tqdm"),
        ("requests",        "requests"),
    ]

    passed = 0
    failed = 0

    for display_name, import_name in libraries:
        if check_import(display_name, import_name):
            passed += 1
        else:
            failed += 1

    print("-"*50)
    print(f"\n📊 Results: {passed} passed | {failed} failed")

    if failed == 0:
        print("\n🎉 ALL LIBRARIES INSTALLED! Ready for Phase 2!")
    else:
        print(f"\n⚠️  Fix {failed} missing libraries before proceeding.")

    print("="*50 + "\n")

if __name__ == "__main__":
    main()