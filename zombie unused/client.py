import socket

LOCALHOST = "127.0.0.1"
PORT = 8080

try:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.connect((LOCALHOST, PORT))

    while True:
        # Placeholder for client-side game input logic
        server.send(bytes(input("Enter your message: "), 'utf-8'))

        data = server.recv(1024)
        print('Received from the server :', str(data.decode()))

        # Placeholder for client-side game state update logic

        answer = input('Do you want to continue(y/n) :')
        if answer.lower() != 'y':
            break

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    server.close()
    print("Disconnected from server")

# game-specific logic in server.py and game-specific tasks handling in client.py
# Implement game mechanics like character movement, zombie behavior, and environmental interactions.