from dotenv import load_dotenv
load_dotenv()
import argparse
from src.webui.interface import theme_map, create_ui


def main():
    parser = argparse.ArgumentParser(description="Gradio WebUI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Glass", choices=theme_map.keys(), help="Theme to use for the UI")
    args = parser.parse_args()

    demo = create_ui(theme_name=args.theme)
    demo.queue().launch(server_name=args.ip, server_port=args.port)


if __name__ == '__main__':
    main()



# theme_map = {
#     "Default": gr.themes.Default(),
#     "Soft": gr.themes.Soft(),
#     "Monochrome": gr.themes.Monochrome(),
#     "Glass": gr.themes.Glass(),
#     "Origin": gr.themes.Origin(),
#     "Citrus": gr.themes.Citrus(),
#     "Ocean": gr.themes.Ocean(),
#     "Base": gr.themes.Base()
# }