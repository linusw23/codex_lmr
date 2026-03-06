from db_storage import install_bootstrap


if __name__ == "__main__":
    install_bootstrap(auto_bootstrap=True)
    print("Database bootstrap complete.")
