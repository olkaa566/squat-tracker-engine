**About the project 🐱**

A high-performance, (soon to be) full-stack application designed to analyze squat depth and biomechanical form. Powered by a custom, dependency-free C++ vision engine, it will track joint kinematics (hip, knee, ankle) to evaluate the depth of the hole in squat and provide immediate form feedback.

Currently in the semi-automatic phase (just started 😶😶), in a few months it will be a fully autonomous movement recognition system.

**Vision/Plans**

* **Currently:** 
User uploads a video, system pauses on the first frame. The user has to manually mark important joints (knee, hip, ankle). Then the system tracks these points frame-by-frame using block-matching algorithms to evaluate the squat.

* **In the future:** 
The manual selection step will be entirely replaced. The system will leverage movement pattern recognition to automatically identify the user, isolate the joints, and track the entire squat movement with zero manual input.

🛠 **Tech Stack**

**Core Vision Engine:**
* **Language:** C++17
* **Image I/O:** `stb_image` (C library for memory-safe image decoding)
* **Build System:** CMake
* **Python Bindings:** Compiled `.pyd` extensions via Python C-API, only for the current state

**Full-Stack Application (In Progress):**
* **Frontend:** Flutter (Cross-platform mobile and web UI)
* **Backend API:** Node.js with TypeScript
* **Database & Storage:** PostgreSQL / Supabase (Handling user kinematics data and video storage)

⚙️ **How to Build the Core Engine**
Clone the repo and build the standalone C++ engine locally:

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```


📄 **License**
This project utilizes third-party libraries (such as stb_image for initial I/O testing) licensed under the MIT License. See THIRD-PARTY-NOTICES.txt for details.