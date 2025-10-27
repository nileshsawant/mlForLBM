# Physics-Aware Neural Network Architecture Diagram

## Complete Architecture Flow

```
INPUT STAGE
┌─────────────────────────────────────┐    ┌─────────────────────────────┐
│           GEOMETRY INPUT            │    │       PARAMETER INPUT       │
│        (60, 40, 30, 1)             │    │           (4,)              │
│                                     │    │                             │
│  3D crack geometry binary mask      │    │  [viscosity, temperature,   │
│  1 = fluid, 0 = solid              │    │   thermal_diffusivity,      │
│                                     │    │   normalized_time]          │
└─────────────────────────────────────┘    └─────────────────────────────┘
                │                                          │
                ▼                                          ▼

ENCODER STAGE - SPATIAL FEATURE EXTRACTION
┌─────────────────────────────────────┐    ┌─────────────────────────────┐
│         GEOMETRY ENCODER            │    │      PARAMETER ENCODER      │
│                                     │    │                             │
│ Conv3D(16, (3,3,3)) + BatchNorm    │    │ Dense(16) + ReLU            │
│         + ReLU                      │    │ Dense(32) + ReLU            │
│         ↓                          │    │ Dropout(0.1)                │
│ Conv3D(24, (3,3,3)) + BatchNorm    │    │         ↓                   │
│         + ReLU                      │    │ Output: (32,) features      │
│ MaxPool3D(2,2,2) → (30,20,15)      │    │                             │
│         ↓                          │    └─────────────────────────────┘
│ Conv3D(32, (3,3,3)) + BatchNorm    │                    │
│         + ReLU                      │                    │
│         ↓                          │                    │
│ Conv3D(32, (3,3,3)) + BatchNorm    │                    │
│         + ReLU                      │                    │
│ MaxPool3D(2,2,2) → (15,10,7)       │                    │
│         ↓                          │                    │
│ GlobalAvgPool3D → (32,)             │                    │
│ Dense(64) + ReLU + Dropout          │                    │
│         ↓                          │                    │
│ Output: (64,) global features       │                    │
└─────────────────────────────────────┘                    │
                │                                          │
                └──────────────┬───────────────────────────┘
                               ▼

FUSION STAGE
┌─────────────────────────────────────────────────────────────────────┐
│                          FEATURE FUSION                            │
│                                                                     │
│  Concatenate([geometry_features(64), parameter_features(32)])       │
│                              ↓                                     │
│                    Combined: (96,) features                        │
│                              ↓                                     │
│                   Dense(128) + ReLU + Dropout                      │
│                              ↓                                     │
│              Dense(15×10×7×16) + Reshape → (15,10,7,16)            │
└─────────────────────────────────────────────────────────────────────┘
                               ▼

DECODER STAGE - SPATIAL RECONSTRUCTION
┌─────────────────────────────────────────────────────────────────────┐
│                      SPATIAL DECODER                               │
│                                                                     │
│ Conv3DTranspose(32, (3,3,3)) + BatchNorm + ReLU                    │
│ UpSampling3D(2,2,2) → (30,20,14)                                   │
│                              ↓                                     │
│ Conv3D(24, (3,3,3)) + BatchNorm + ReLU                             │
│ UpSampling3D(2,2,2) → (60,40,28)                                   │
│                              ↓                                     │
│ ZeroPadding3D → (60,40,30)  [Back to original resolution]          │
│                              ↓                                     │
│ Conv3D(16, (3,3,3)) + BatchNorm + ReLU                             │
│                              ↓                                     │
│              SHARED_FEATURES: (60,40,30,16)                        │
└─────────────────────────────────────────────────────────────────────┘
                               ▼

PHYSICS-AWARE OUTPUT STAGE
┌─────────────────────────────────────────────────────────────────────┐
│                    PHYSICS-AWARE OUTPUTS                           │
│                                                                     │
│            SHARED_FEATURES (60,40,30,16)                           │
│                         │                                          │
│        ┌────────────────┼────────────────┬─────────────────────────┐│
│        │                │                │                         ││
│        ▼                ▼                ▼                         ││
│                                                                     │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│ │ Conv3D(3,   │  │ Conv3D(3,   │  │ Conv3D(1,   │                  │
│ │ (1,1,1),    │  │ (1,1,1),    │  │ (1,1,1),    │                  │
│ │ tanh)       │  │ tanh)       │  │ softplus)   │                  │
│ │             │  │             │  │             │                  │
│ │ VELOCITY    │  │ HEAT_FLUX   │  │ DENSITY     │                  │
│ │ (60,40,30,3)│  │ (60,40,30,3)│  │ (60,40,30,1)│                  │
│ │ Range:[-1,1]│  │ Range:[-1,1]│  │ Range:[0,∞) │                  │
│ └─────────────┘  └─────────────┘  └─────────────┘                  │
│                                                                     │
│        ┌─────────────┐            ┌─────────────┐                  │
│        │ Conv3D(1,   │            │ Conv3D(1,   │                  │
│        │ (1,1,1),    │            │ (1,1,1),    │                  │
│        │ softplus)   │            │ softplus)   │                  │
│        │             │            │             │                  │
│        │ ENERGY      │            │TEMPERATURE  │                  │
│        │ (60,40,30,1)│            │ (60,40,30,1)│                  │
│        │ Range:[0,∞) │            │ Range:[0,∞) │                  │
│        └─────────────┘            └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Architecture Insights

### 1. **Encoder Stage: What Conv3D(3,3,3) Does**
- **Local Pattern Recognition**: Each 3×3×3 kernel examines 27 neighboring voxels
- **Spatial Context**: Learns relationships like "crack wall → low velocity nearby"
- **Hierarchical Features**: 
  - Layer 1: Edges and boundaries
  - Layer 2: Local flow patterns  
  - Layer 3: Regional flow structures
  - Layer 4: Global connectivity patterns

### 2. **Parameter Integration**
- Physics parameters (viscosity, temperature, etc.) are processed separately
- Fused with geometry features to create physics-aware representations
- This allows the same crack geometry to produce different flows for different physics

### 3. **Decoder Stage: Spatial Reconstruction**  
- Takes compressed features and rebuilds full 3D spatial resolution
- Conv3DTranspose and UpSampling restore (60,40,30) dimensions
- Final Conv3D(16, (3,3,3)) creates rich spatial features for output branches

### 4. **Physics-Aware Outputs: Why Conv3D(1,1,1)**
- **Input**: Rich 16-channel spatial features at each voxel location
- **Operation**: Pointwise transformation (no spatial mixing)
- **Purpose**: Convert features → physics fields with constraints
- **Result**: Each voxel gets independent physics-compliant values

## The Magic of Conv3D(1,1,1)

Think of it this way:
```
At every voxel (i,j,k):
Input:  [16 rich spatial features] ← From earlier Conv3D(3,3,3) layers
        ↓
Conv3D(1,1,1) + Activation:
        ↓
Output: [1 physics field value] ← With mathematical constraints

The (1,1,1) kernel is like a "smart lookup table":
- Takes spatial understanding → Produces physics-compliant values
- No neighbor mixing → Pure pointwise constraint application
```

This design is brilliant because:
- **Spatial relationships**: Handled by early Conv3D(3,3,3) layers
- **Physics constraints**: Applied by final Conv3D(1,1,1) + activation layers
- **Efficiency**: Best of both worlds - spatial awareness + guaranteed physics compliance