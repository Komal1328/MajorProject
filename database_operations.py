import mysql.connector
from mysql.connector import Error
import datetime
import os


def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='smartflow',
            user='root',
            password=''
        )
        if connection.is_connected():
            print("Connection to MySQL database was successful")
        return connection
    except Error as e:
        print(f"Error: {e}")
        return None


def create_table(connection):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS images (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        path VARCHAR(255) NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    cursor = connection.cursor()
    cursor.execute(create_table_query)
    connection.commit()

def create_ambulance_table(connection):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS ambulance_images (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        path VARCHAR(255) NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    cursor = connection.cursor()
    cursor.execute(create_table_query)
    connection.commit()


def insert_image(connection, image_path):
    if not os.path.exists(image_path):
        print(f"Error: The file {image_path} does not exist.")
        return

    cursor = connection.cursor()
    with open(image_path, 'rb') as file:
        binary_data = file.read()

    date_today = datetime.date.today()
    time_now = datetime.datetime.now().time()
    extension = os.path.splitext(image_path)[1][1:]

    insert_query = """
    INSERT INTO traffic_images (image, date, time, extension)
    VALUES (%s, %s, %s, %s)
    """
    cursor.execute(insert_query, (binary_data, date_today, time_now, extension))
    connection.commit()
    image_id = cursor.lastrowid

    image_name = f"{image_id}.{extension}"
    update_query = """
    UPDATE traffic_images
    SET name = %s
    WHERE id = %s
    """
    cursor.execute(update_query, (image_name, image_id))
    connection.commit()

    cursor.close()
    print(f"Image inserted successfully with ID {image_id} and name {image_name}")


def retrieve_recent_image(connection, save_path):
    cursor = connection.cursor()
    select_query = "SELECT name, image, extension FROM traffic_images ORDER BY id DESC LIMIT 1"
    cursor.execute(select_query)
    record = cursor.fetchone()
    if record:
        name, image_data, extension = record
        file_name = f"{name}"
        file_path = os.path.join(save_path, file_name)
        with open(file_path, 'wb') as file:
            file.write(image_data)
        print(f"Most recent image {file_name} saved to {save_path}")
        cursor.close()
        return file_path
    else:
        print("No images found")
        cursor.close()
        return None


# insert image  to data base

def insert_ambulance_image(connection, image_path):
    if not os.path.exists(image_path):
        print(f"Error: The file {image_path} does not exist.")
        return

    cursor = connection.cursor()
    with open(image_path, 'rb') as file:
        binary_data = file.read()

    date_today = datetime.date.today()
    time_now = datetime.datetime.now().time()
    extension = os.path.splitext(image_path)[1][1:]  # Get the file extension without the dot

    # Insert the image and get the inserted ID
    insert_query = """
    INSERT INTO ambulance_images (image, date, time, extension)
    VALUES (%s, %s, %s, %s)
    """
    cursor.execute(insert_query, (binary_data, date_today, time_now, extension))
    connection.commit()
    image_id = cursor.lastrowid  # Get the ID of the inserted row

    # Update the image name with the ID
    image_name = f"{image_id}.{extension}"
    update_query = """
    UPDATE ambulance_images
    SET name = %s
    WHERE id = %s
    """
    cursor.execute(update_query, (image_name, image_id))
    connection.commit()

    cursor.close()
    print(f"Image inserted successfully with ID {image_id} and name {image_name}")


# retrieve image from database

def retrieve_ambulance_recent_image(connection, save_path):
    cursor = connection.cursor()
    select_query = "SELECT name, image, extension FROM ambulance_images ORDER BY id DESC LIMIT 1"
    cursor.execute(select_query)
    record = cursor.fetchone()
    if record:
        name, image_data, extension = record
        file_name = f"{name}"
        file_path = os.path.join(save_path, file_name)
        with open(file_path, 'wb') as file:
            file.write(image_data)
        print(f"Most recent image {file_name} saved to {save_path}")
        cursor.close()
        return file_path
    else:
        print("No images found")
        cursor.close()
        return None
