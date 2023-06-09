import argparse
import panel as pn
from .panel_dashboard import create_dashboard
from .ml_worker import MLWorker
import logging

def main():
    parser = argparse.ArgumentParser(description='Serve a Panel application.')
    parser.add_argument('-p', '--port', default=5006, type=int, help='Port to serve the application on')
    parser.add_argument('-a', '--address', default='0.0.0.0', type=str, help='Address to bind the application to')
    parser.add_argument('-s', '--show', default=False, action='store_true', help='Open the server in a browser after startup')
    parser.add_argument('-w', '--websocket-origin', nargs='*', default=None, help='Hosts that can access the websocket')
    args = parser.parse_args()
    routes = {
        '/': create_dashboard,
    }

    # set logging level to info
    logging.basicConfig(level=logging.INFO)
    
    logging.info(f"Starting server on {args.address}:{args.port}")
    pn.serve(routes, port=args.port, address=args.address, show=args.show, 
             websocket_origin=args.websocket_origin)
    
    logging.info('Stopping MLWorker')

    worker = MLWorker()
    worker.stop()


if __name__ == '__main__':
    main()
