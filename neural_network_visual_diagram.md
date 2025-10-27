# Physics-Aware Neural Network - Visual Architecture Diagram

## Left-to-Right Neural Network Flow

```
INPUT LAYER              ENCODER LAYERS (Geometry Only)                    BOTTLENECK
                                                                           
  Geometry                                                                    Global
  (60×40×30×1)           Conv3D Layers with (3×3×3) Kernels               Features
       ●                                                                      ●
       ●                 ████████████████        ████████████████             ●
       ●                 ████████████████        ████████████████             ●
       ●       ────────► ████████████████  ────► ████████████████  ────────►  ●
       ●                 ████████████████        ████████████████             ●
       ●                 ████████████████        ████████████████             ●
       ●                      16 ch                   24 ch                  ●
       ●                                                                      ●
                              │                        │                     
  Parameters                  │                        │                     
     (4×1)                    ▼                        ▼                     
       ●                 MaxPool3D                MaxPool3D                  
       ●                 (30×20×15)               (15×10×7)                  
       ●        ┌─────────────────────────────────────────────┐              
       ●        │                                             │              
              Dense                                        Dense            
              Layers                                       (64)             
               (32)                                          ●              
                ●                                           ●               
                ●                                                            
                └─────────────► FUSION ◄─────────────────────┘              


DECODER LAYERS (Geometry + Physics)                     OUTPUT LAYERS (Physics-Aware)

     Fused                                                   Final Fields
   Features              Spatial Reconstruction              (60×40×30×N)
      ●                                                            
      ●               ████████████████████                   ┌─── Velocity (3) ───┐
      ●    ────────►  ████████████████████  ────────────────┤                     │
      ●               ████████████████████                   │ ┌─ Conv3D(1×1×1) ──┤
      ●               ████████████████████                   │ │    + tanh        │
     (128)                  32 ch                           │ └─ [-1, +1] range  │
                              │                             │                     │
                              ▼                             ├─── Heat Flux (3) ──┤
                         UpSample3D                         │                     │
                         (30×20×14)                         │ ┌─ Conv3D(1×1×1) ──┤
                              │                             │ │    + tanh        │
                              ▼                             │ └─ [-1, +1] range  │
                      ████████████████                      │                     │
                      ████████████████                      ├─── Density (1) ────┤
                      ████████████████  ──────────────────► │                     │
                      ████████████████                      │ ┌─ Conv3D(1×1×1) ──┤
                      ████████████████                      │ │   + softplus     │
                            24 ch                           │ └─ [0, ∞) range    │
                              │                             │                     │
                              ▼                             ├─── Energy (1) ─────┤
                         UpSample3D                         │                     │
                         (60×40×28)                         │ ┌─ Conv3D(1×1×1) ──┤
                              │                             │ │   + softplus     │
                              ▼                             │ └─ [0, ∞) range    │
                      ████████████████                      │                     │
                      ████████████████                      ├─ Temperature (1) ──┤
                      ████████████████  ──────────────────► │                     │
                      ████████████████                      │ ┌─ Conv3D(1×1×1) ──┤
                      ████████████████                      │ │   + softplus     │
                            16 ch                           │ └─ [0, ∞) range    │
                        SHARED FEATURES                     └─────────────────────┘
                        (60×40×30×16)                           PHYSICS-AWARE
                                                                 CONSTRAINTS
```

## Layer Width Representation

### Input Stage
```
Geometry Input:     ████████████████████████████████████████ (72,000 voxels)
Parameters Input:   ████ (4 parameters)
```

### Encoder Stage (Geometry Learning)
```
Conv3D(16):        ████████████████ (16 channels × spatial)
Conv3D(24):        ████████████████████████ (24 channels × spatial)  
Conv3D(32):        ████████████████████████████████ (32 channels × spatial)
Conv3D(32):        ████████████████████████████████ (32 channels × spatial)
Global Pool:       ████████████████ (64 global features)
```

### Parameter Processing
```
Dense(16):         ████████ (16 features)
Dense(32):         ████████████████ (32 features)
```

### Fusion Stage
```
Concatenated:      ████████████████████████ (96 combined features)
Dense(128):        ████████████████████████████████ (128 fused features)
```

### Decoder Stage (Physics-Aware Learning)
```
Reshape:           ████████████████████████████████ (15×10×7×16)
Conv3DTranspose:   ████████████████████████████████ (32 channels)
Conv3D(24):        ████████████████████████ (24 channels)
Conv3D(16):        ████████████████ (16 shared features)
```

### Output Stage (Physics Constraints)
```
Each Conv3D(1×1×1) Output:
Velocity:    ███ (3 channels, tanh bounded)
Heat Flux:   ███ (3 channels, tanh bounded)  
Density:     █ (1 channel, softplus positive)
Energy:      █ (1 channel, softplus positive)
Temperature: █ (1 channel, softplus positive)
```

## Information Flow Summary

```
INPUT           ENCODER          FUSION           DECODER          OUTPUT
─────           ───────          ──────           ───────          ──────

Geometry  ──►   Learn Pure   ──► Combine     ──► Learn Physics ──► Apply Physics
Binary          Geometric        Geometry         + Geometry       Constraints
Mask            Features         + Physics        Relationships    Pointwise
                                 Context                           
Parameters ──►  Learn Physics ──►                                  
Values          Context                                            

Width:     Large ──► Medium ──► Small ──► Medium ──► Large ──► 5 Fields
Channels:  1 ────► 16→24→32 ─► 128 ──► 32→24→16 ─► 1 or 3 per field
Resolution: Full ─► Reduced ──► Global ─► Restored ─► Full ──► Full
```

## Key Architectural Insights

### 1. **Hourglass Shape**: 
   - Wide input (72K voxels) → Narrow bottleneck (128 features) → Wide output (72K×5 fields)

### 2. **Separation of Concerns**:
   - **Left side**: Geometry understanding only
   - **Middle**: Geometry + physics fusion  
   - **Right side**: Physics-aware field generation

### 3. **Physics Constraints Applied Last**:
   - All spatial learning happens with (3×3×3) kernels
   - Physics constraints applied with (1×1×1) kernels + smart activations
   - No mixing of spatial information during constraint application

This architecture elegantly separates geometric understanding from physics constraint enforcement, leading to both accurate predictions and guaranteed physics compliance!