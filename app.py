from ui.app_ui import create_app


demo = create_app()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
