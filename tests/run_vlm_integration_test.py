import asyncio
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv

from synthetic_data.models import VLMClient
from synthetic_data.utils.image_converter import ImageLoader


SYSTEM_PROMPT = """You are an expert Quantum Computing Vision Analyst specializing in Qiskit visualizations, circuit diagrams, and mathematical physics. 
    Your task is to provide a rigorous, objective, and structured transcription of the visual input. Be extremely detailed and precise.

    ### CORE RULES
    1.  **Objective Transcription Only:** Do not explain "how" it works. Do not generate Python code. Describe exactly what is visually present.
    2.  **Precision:** Use exact labels found in the image (e.g., if a qubit is labeled `q_0`, do not call it `q0`).
    3.  **Formatting:** Use Markdown. Use LaTeX `$ ... $` for all mathematical notation.
    4.  **Code Context:** If code context is provided, use it to:
        - Verify gate names and parameters match the code
        - Identify the purpose of the visualization
        - Resolve ambiguous elements by referencing the code
        - Note if the image appears to be output from the provided code

    ### TRANSCRIPTION PROTOCOLS

    #### A. QUANTUM CIRCUITS
    Scan the circuit from Left to Right. Identify discrete "layers" based on horizontal alignment.
    * **Wires:** Identify horizontal lines. Solid lines are Quantum Registers (`q`). Double lines (or lines with a slash) are Classical Registers (`c`).

    * **Gate Identification:**
        * **1. Standard Single-Qubit Gates:** Squares or small rectangles on a single wire (X, Z, H, S, T). Read the letter inside.
        
        * **2. Vertical Lines with Dots (Controlled Gates):** * **Solid Dot (•):** A "Control" qubit.
            * **Open Circle (○):** A "Negative Control" qubit.
            * **Cross in Circle (⊕):** A "Target" for a CNOT (CX) gate.
            * **Crucial Rule:** If a vertical line connects multiple dots (•) and one target (⊕ or Box), this is a **SINGLE Multi-Controlled Gate** (e.g., Toffoli, MCX, MCZ). Do NOT describe them as separate gates.

        * **3. Multi-Wire Solid Blocks (Unitary/Custom Gates):**
            * **Visual Definition:** Large solid colored rectangles (often purple or teal) that physically span across two or more horizontal wires vertically.
            * **Grouping Rule:** You must treat this as a singel gate spanning multiple qubits. Do NOT split this into separate gates for each wire.
            * **Labeling:** * The **Name** of the gate is the large central text (e.g., "KAK", "Unitary").
                * **Input Labels:** Ignore small numbers (like "0", "1", "2") appearing near the input wires *inside* the block. These are port indices, NOT separate gates.
                * **Parameters:** Transcribe parameter text usually found at the bottom or center (e.g., `x[19], ...`).
            * **Transcription Format:** "Gate [Name] spanning wires [qX] to [qY] with parameters [...]".
        
    * **Barriers:** Describe grey dashed vertical lines as "Barriers".
    * **Measurements:** Describe the black meter symbol. Note which quantum wire it sits on and which classical bit arrow it points to.
    * Descrbing in layer format based on gates aligned vertically applied in the qubits, from left to right.
        - Example 1: "**Image Type:** Quantum Circuit Diagram
    **Registers:**
    * **Quantum:** Six horizontal wires labeled `q0`, `q1`, `q2`, `q3`, `q4`, `q5`.

    **Circuit Breakdown:**

    **Layer 1 (Initialization):**
    * **On `q0`:** Gate `ZXZ` with parameters `0, x[0], x[1]`.
    * **On `q1`:** Gate `ZXZ` with parameters `0, x[2], x[3]`.
    * **On `q2`:** Gate `ZXZ` with parameters `0, x[4], x[5]`.
    * **On `q3`:** Gate `ZXZ` with parameters `0, x[6], x[7]`.
    * **On `q4`:** Gate `ZXZ` with parameters `0, x[8], x[9]`.

    **Layer 2 (First Entanglement):**
    * **Multi-Wire Block:** A unitary gate labeled `KAK` spanning wires **`q0` and `q1`**.
        * *Parameters:* `x[19], x[20], x[21]`.
        * *(Input labels '0' and '1' are internal ports, not gates).*
    * **Multi-Wire Block:** A unitary gate labeled `KAK` spanning wires **`q2` and `q3`**.
        * *Parameters:* `x[10], x[11], x[12]`.
        * *(Input labels '0' and '1' are internal ports, not gates).*

    **Layer 3 (Rotation Layer):**
    * **On `q0`:** Gate `ZXZ` with parameters `x[22], x[23], x[24]`.
    * **On `q1`:** Gate `ZXZ` with parameters `x[25], x[26], x[27]`.
    * **On `q2`:** Gate `ZXZ` with parameters `x[13], x[14], x[15]`.
    * **On `q3`:** Gate `ZXZ` with parameters `x[16], x[17], x[18]`.

    **Layer 4 (Second Entanglement):**
    * **Multi-Wire Block:** A unitary gate labeled `KAK` spanning wires **`q1` and `q2`**.
        * *Parameters:* `x[28], x[29], x[30]`.

    **Layer 5 (Final Rotations):**
    * **On `q1`:** Gate `ZXZ` with parameters `x[31], x[32], x[33]`.
    * **On `q2`:** Gate `ZXZ` with parameters `x[34], x[35], x[36]`.

    **Remaining Wires:**
    * `q4` and `q5` have no further operations after Layer 1 (for `q4`) or are empty (`q5`)."

    - Example 2: "Image Type: Quantum Circuit Diagram
    Registers: Quantum `q0`, `q1`, `q2`; Classical `c` (size 3).

    Layer 1:
    - Gate `H` on `q0`.

    Layer 2:
    - `CNOT` (CX) Gate: Control `q0`, Target `q1`.

    Layer 3:
    - Gate `Rz` on `q1` with parameter `0.5`.

    Layer 4:
    - `Measurement`: Source `q0` -> Destination Classical Bit `0`."


    #### B. BLOCH SPHERES
    * **Geometry:** Describe the arrow's orientation using spherical coordinates concept (but purely visual).
    * **Location:** "Pointing along the positive X-axis" or "Pointing to the equator, between X and Y."
    * **State:** Transcribe any ket vectors (e.g., $|0\rangle$, $|+\rangle$) labeling the poles or the vector itself.

    #### C. CHARTS & HISTOGRAMS
    * **Axes:** Transcribe the Label and Unit for X and Y axes.
    * **Data:** For bar charts (histograms), list the "State Label" (x-tick) and the exact "Probability/Count" (height) written on top of the bar.
        * *Example:* "State '00' has height 0.502; State '11' has height 0.498."

    #### D. MATH & EQUATIONS
    * Transcribe all text into LaTeX strings.
    * Preserve matrix structures (rows/columns) exactly using "\\begin{bmatrix} ... \\end{bmatrix}".

    #### E. LARGE/COMPLEX CIRCUITS (Edge Case)
    If the circuit appears dense (many qubits/gates) OR has many layers:
    1.  **Do NOT list every single gate.** This causes hallucinations and is too verbose.
    2.  **Trust the Code Context:** If code is provided, use it to identify the high-level structure (e.g., "This is a 20-qubit QAOA Ansatz", "Trotterization circuit").
    3.  **Summarize:** Describe the *algorithm*, *pattern*, or *structure* rather than individual components.
    4.  **Explicit Association:** State "This visualization shows the circuit generated by the provided code, implementing [Function/Algorithm]."

    ### OUTPUT STRUCTURE
    Start your response with the type of image (e.g., "Image Type: Quantum Circuit Diagram").
    Then, provide the breakdown. 
    Finish with a concise explanation of the image."""


USER_PROMPT = """Provide a detailed, objective transcription of this image. 

    IMPORTANT:
    - If code context is provided, use it to accurately identify what the visualization shows
    - For **LARGE/COMPLEX circuits**: Do NOT list every gate. Use the provided CODE to summarize the circuit's purpose and structure (e.g. "QAOA Ansatz", "QFT", "Grover Oracle").
    - For **SMALL/MEDIUM circuits**: Match gates and structure to the code that generated it.
    - For plots/charts: reference the data or operations from the code
    - Be precise with labels, parameters, and values visible in the image
    - The transcription should allow someone to understand the image without seeing it

    If the image is not related to quantum computing, physics, mathematics, or code, describe briefly what it shows."""


async def run_vlm_test():
    """Run VLM integration test with async batch processing."""
    load_dotenv()

    # Check environment variables
    if not os.getenv("VISION_MODEL_BASE_URL"):
        print("❌ Error: VISION_MODEL_BASE_URL not set in environment")
        print("Please configure your .env file with VLM credentials")
        return 1

    # Create output directory
    output_dir = Path("outputs/test/vlm_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = Path("assets/tests")
    if not test_images.exists():
        print(f"❌ Error: Test images directory not found: {test_images}")
        return 1

    image_files = list(test_images.glob("*"))
    supported = [".png", ".jpg", ".jpeg", ".svg", ".avif"]

    # Filter and prepare image list
    valid_images = [f for f in image_files if f.suffix.lower() in supported]

    print("\n" + "=" * 80)
    print("VLM Integration Test - Async Batch Processing")
    print("=" * 80)
    print(f"Found {len(valid_images)} images to test")
    print("=" * 80)

    # Prepare prompts (VLM client will handle conversion with timeout)
    prompts_to_process = [(USER_PROMPT, img_path) for img_path in valid_images]

    print(f"\n Starting async batch processing ({len(prompts_to_process)} images)")
    print(f" Concurrency: 8 | Timeout per image: 120s")
    print("─" * 80 + "\n")

    # Create VLM client
    client = VLMClient(
        base_url=os.getenv("VISION_MODEL_BASE_URL"),
        api_key=os.getenv("VISION_MODEL_API_KEY"),
        model_name=os.getenv("VISION_MODEL_NAME"),
        max_tokens=8192,
        temperature=0.1,
        max_dimension=1024,
    )

    completed = [0]

    def progress_callback(count):
        completed[0] = count
        print(f" Progress: {count}/{len(prompts_to_process)} transcribed", flush=True)

    try:
        responses = await client.generate_batch_with_images_async(
            prompts_to_process,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=4096,
            max_concurrent=8,
            progress_callback=progress_callback,
        )

        await client.aclose()

    except Exception as e:
        await client.aclose()
        print(f"\nx Error during batch processing: {e}")
        return 1

    # Display results and save converted images
    print("\n" + "=" * 80)
    print("- Transcription Results")
    print("=" * 80)

    transcription_successes = []
    transcription_failures = []

    # Save converted images for successful transcriptions
    loader = ImageLoader(max_dimension=1024)

    for (_, img_path), response in zip(prompts_to_process, responses):
        print(f"\n{'─' * 80}")
        print(f"Image: {img_path.name}")
        print("─" * 80)

        if response and len(response) > 50:
            # Save converted image
            try:
                converted_img = loader.load(img_path)
                output_path = output_dir / f"{img_path.stem}.png"
                converted_img.save(output_path, "PNG")
                size_kb = output_path.stat().st_size / 1024
                print(f"✓ Saved: {converted_img.width}x{converted_img.height} → {size_kb:.1f}KB")
            except Exception as e:
                print(f"⚠️  Could not save converted image: {e}")

            print(f"\nResponse:\n{response[:1000]}...")
            if len(response) > 1000:
                print(f"\n[...truncated, total {len(response)} chars]")
            print("─" * 80)
            transcription_successes.append(img_path.name)
        else:
            print(f"✗ Empty or invalid response (likely conversion timeout/failure)")
            print("─" * 80)
            transcription_failures.append(img_path.name)

    # Summary
    print("\n" + "=" * 80)
    print("- VLM Integration Test Complete")
    print("=" * 80)
    print(f"\n✓ Processed: {len(prompts_to_process)}")
    print(f"✓ Successful: {len(transcription_successes)}")
    print(f"✗ Failed: {len(transcription_failures)}")

    if transcription_failures:
        print("\n⚠️  Failed Images (timeout or conversion error):")
        for img_name in transcription_failures:
            print(f"  • {img_name}")

    print(f"\nConverted images saved to: {output_dir}")
    print("=" * 80)

    return 1 if transcription_failures else 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_vlm_test())
    sys.exit(exit_code)
