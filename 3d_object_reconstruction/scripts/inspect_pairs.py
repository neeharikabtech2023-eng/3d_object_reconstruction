import sqlite3
from pathlib import Path

DB_PATH = Path(
    r"C:\Users\ganga\Downloads\3d_object_reconstruction_project"
    r"\3d_object_reconstruction\projects\my_first_reconstruction\database.db"
)

MAX_IMAGE_ID = 2147483647

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

connection = sqlite3.connect(str(DB_PATH))

images = {
    image_id: name
    for image_id, name in connection.execute(
        "SELECT image_id, name FROM images"
    )
}

query = """
SELECT pair_id, rows
FROM two_view_geometries
WHERE rows > 0
ORDER BY rows DESC
LIMIT 20
"""

results = connection.execute(query).fetchall()

print("\nTOP 20 GEOMETRICALLY VERIFIED IMAGE PAIRS\n")

for pair_id, inliers in results:
    image_id1 = pair_id // MAX_IMAGE_ID
    image_id2 = pair_id % MAX_IMAGE_ID

    name1 = images.get(image_id1, "Unknown image")
    name2 = images.get(image_id2, "Unknown image")

    print(f"Inliers: {inliers:5d}")
    print(f"  Image 1: {name1} (ID {image_id1})")
    print(f"  Image 2: {name2} (ID {image_id2})")
    print("-" * 60)

connection.close()
