import socket
import threading
import time


class ClientThread(threading.Thread):
    def __init__(self, clientAddress, clientsocket):
        threading.Thread.__init__(self)
        self.csocket = clientsocket
        print("New connection added: ", clientAddress)

    def run(self):
        print("Connection from : ", clientAddress)
        # Placeholder for server-side game initialization logic

        try:
            self.csocket.send(bytes("Hi, This is from Server..",'utf-8'))
            while True:
                try:
                    data = self.csocket.recv(2048)
                except ConnectionResetError:
                    print("Client at ", clientAddress , " disconnected unexpectedly.")
                    break

                msg = data.decode()
                if msg == 'bye':
                    print("Client at ", clientAddress , " requested to disconnect.")
                    break

                print("From client", msg)

                # Placeholder for server-side game update logic

                self.csocket.send(bytes(msg,'UTF-8'))
        except Exception as e:
            print(f"An error occurred with client at {clientAddress}: {e}")
        finally:
            print(f"Closing connection with {clientAddress}")
            self.csocket.close()

LOCALHOST = "127.0.0.1"
PORT = 8080

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((LOCALHOST, PORT))
server.listen(1)
server.settimeout(1.0)  # Set timeout for the accept call

print("Server started")
print("Waiting for client request..")

running = True

try:
    while running:
        try:
            clientsock, clientAddress = server.accept()
            newthread = ClientThread(clientAddress, clientsock)
            newthread.start()
        except socket.timeout:
            # Check for stop flag every second
            continue
        except KeyboardInterrupt:
            print("Server is shutting down...")
            running = False
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    server.close()
