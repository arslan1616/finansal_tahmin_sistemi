import websockets
import asyncio
import json

async def fetch_realtime_data():
    """
    WebSocket kullanarak gerçek zamanlı hisse senedi verilerini çeker.
    """
    uri = "wss://stream.binance.com:9443/ws/garantis@trade"  # Örnek WebSocket URI
    async with websockets.connect(uri) as websocket:
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            print(data)  # Gerçek zamanlı veriyi işle

# WebSocket'i başlat
if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(fetch_realtime_data())