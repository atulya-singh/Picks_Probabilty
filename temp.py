import kagglehub

# Download latest version
path = kagglehub.dataset_download("cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024")

print("Path to dataset files:", path)