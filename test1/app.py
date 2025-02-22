import subprocess
import argparse
import os

def change_vllm_image(container_name, new_image, model_path, port, gpu_ids="all"):
    """
    Changes the vLLM container image by stopping the existing container,
    pulling the new image, and starting a new container with the specified settings.

    Args:
        container_name (str): The name of the existing vLLM container.
        new_image (str): The name and tag of the new Docker image.
        model_path (str): The path to the model directory on the host machine.
        port (int): The port to expose for the vLLM service.
        gpu_ids (str, optional): The GPU IDs to use (e.g., "all", "0,1"). Defaults to "all".
    """

    try:
        # Stop the existing container
        subprocess.run(["docker", "stop", container_name], check=True)
        print(f"Container '{container_name}' stopped.")

        # Pull the new image
        subprocess.run(["docker", "pull", new_image], check=True)
        print(f"Image '{new_image}' pulled.")

        # Extract model name from path.
        model_name = os.path.basename(model_path)

        # Construct the docker run command
        docker_run_command = [
            "docker", "run",
            "--runtime", "nvidia",
            "--gpus", gpu_ids,
            "-p", f"{port}:{port}",
            "-v", f"{model_path}:/app/model",
            "--name", container_name,
            new_image,
            "--model", f"/app/model/{model_name}"
        ]

        # Run the new container
        subprocess.run(docker_run_command, check=True)
        print(f"Container '{container_name}' started with image '{new_image}'.")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print("Error: Docker not found. Make sure Docker is installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change vLLM container image.")
    parser.add_argument("container_name", help="Name of the vLLM container.")
    parser.add_argument("new_image", help="Name and tag of the new Docker image.")
    parser.add_argument("model_path", help="Path to the model directory on the host.")
    parser.add_argument("--port", type=int, default=8000, help="Port to expose (default: 8000).")
    parser.add_argument("--gpu_ids", default="all", help="GPU IDs to use (e.g., 'all', '0,1').")

    args = parser.parse_args()

    change_vllm_image(args.container_name, args.new_image, args.model_path, args.port, args.gpu_ids)

