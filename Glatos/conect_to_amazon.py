import psycopg2  # For PostgreSQL


# For MySQL, you could use: import pymysql

def connect_to_amazon_db(dbname, host, port, username, password):
    """
    Establishes a connection to an Amazon RDS database.

    Args:
        dbname (str): The name of the database to connect to
        host (str): The hostname of the Amazon RDS instance
        port (int): The port number to connect on
        username (str): The username for authentication
        password (str): The password for authentication

    Returns:
        connection: A database connection object
    """
    try:
        # For PostgreSQL
        conn = psycopg2.connect(
            dbname=dbname,
            host=host,
            port=port,
            user=username,
            password=password
        )
        print("Connection to Amazon RDS established successfully!")
        return conn
    except Exception as e:
        print(f"Error connecting to Amazon RDS: {e}")
        return None