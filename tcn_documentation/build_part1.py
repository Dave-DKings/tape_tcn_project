"""
Complete TCN Documentation Builder
Builds all 10 sections with mathematical rigor and references
"""
import json

def build_complete_tcn_docs():
    """Build comprehensive TCN documentation with all sections"""
    
    notebook = json.load(open(r"c:\Users\Owner\new_project\adaptive_portfolio_rl\tcn_documentation\03_tcn_architectures_and_variants_EXPANDED.ipynb"))
    
    # Section 2: Theoretical Foundations
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Theoretical Foundations <a id='section2'></a>\n",
            "\n",
            "### 2.1 Causal Convolution\n",
            "\n",
            "A **causal convolution** ensures output at time $t$ depends only on inputs $\\leq t$, never future timesteps.\n",
            "\n",
            "For 1D convolution with kernel size $k$:\n",
            "\n",
            "$$\n",
            "y_t = \\sum_{i=0}^{k-1} w_i \\cdot x_{t-i}\n",
            "$$\n",
            "\n",
            "Implementation: **causal padding** - left-pad input by $(k-1)$ zeros before standard convolution.\n",
            "\n",
            "### 2.2 Dilated Convolution\n",
            "\n",
            "A **dilated convolution** with rate $d$ samples input with gaps:\n",
            "\n",
            "$$\n",
            "y_t = \\sum_{i=0}^{k-1} w_i \\cdot x_{t - i \\cdot d}\n",
            "$$\n",
            "\n",
            "- $d=1$: standard convolution\n",
            "- $d=2$: samples every other timestep  \n",
            "- $d=4$: samples every 4th timestep\n",
            "\n",
            "**Benefit**: Exponentially increases receptive field without adding parameters.\n",
            "\n",
            "### 2.3 Residual Connections\n",
            "\n",
            "Each TCN block includes skip connection:\n",
            "\n",
            "$$\n",
            "\\text{output} = \\text{Activation}(\\text{Conv}(x) + x)\n",
            "$$\n",
            "\n",
            "If dimensions mismatch, use 1x1 projection:\n",
            "\n",
            "$$\n",
            "\\text{output} = \\text{Activation}(\\text{Conv}(x) + W_{\\text{proj}} \\cdot x)\n",
            "$$\n",
            "\n",
            "**Benefit**: Enables gradient flow through very deep networks (He et al. 2015).\n",
            "\n",
            "### 2.4 TCN vs RNN/LSTM\n",
            "\n",
            "| Aspect | RNN/LSTM | TCN |\n",
            "|--------|----------|-----|\n",
            "| **Training** | Sequential | Fully parallel |\n",
            "| **Speed** | Slow | Fast (GPU-optimized) |\n",
            "| **Receptive Field** | Full history | Controlled by design |\n",
            "| **Gradient Flow** | Vanishing/exploding | Stable (residuals) |\n",
            "| **Memory** | $O(T)$ states | $O(1)$ per step |\n",
            "| **Long Dependencies** | Difficult | Excellent (dilations) |\n",
            "\n",
            "**References**: Bai et al. (2018), He et al. (2015) for ResNets"
        ]
    })
    
    # Section 3: TCN Block Implementation  
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. TCN Block Implementation <a id='section3'></a>\n",
            "\n",
            "### 3.1 TCN Block Structure\n",
            "\n",
            "Each `TCNBlock` contains **two causal dilated Conv1D layers** + **residual connection**.\n",
            "\n",
            "For input sequence $x_t$, kernel size $k$, dilation $d$:\n",
            "\n",
            "**First convolution**:\n",
            "$$\n",
            "y_t^{(1)} = \\text{ReLU}\\left(\\sum_{i=0}^{k-1} W_i^{(1)}x_{t-id} + b^{(1)}\\right)\n",
            "$$\n",
            "\n",
            "**Dropout** (training only):\n",
            "$$\n",
            "\\tilde{y}_t^{(1)} = \\text{Dropout}(y_t^{(1)}, p)\n",
            "$$\n",
            "\n",
            "**Second convolution**:\n",
            "$$\n",
            "y_t^{(2)} = \\text{ReLU}\\left(\\sum_{i=0}^{k-1} W_i^{(2)}\\tilde{y}_{t-id}^{(1)} + b^{(2)}\\right)\n",
            "$$\n",
            "\n",
            "**Dropout** (training only):\n",
            "$$\n",
            "\\tilde{y}_t^{(2)} = \\text{Dropout}(y_t^{(2)}, p)\n",
            "$$\n",
            "\n",
            "**Residual + activation**:\n",
            "$$\n",
            "\\text{out}_t = \\text{ReLU}(\\tilde{y}_t^{(2)} + r_t)\n",
            "$$\n",
            "\n",
            "where $r_t = x_t$ if dimensions match, else $r_t = W_{\\text{proj}} x_t$ (1x1 conv).\n",
            "\n",
            "### 3.2 Implementation Code\n",
            "\n",
            "From `src/agents/actor_critic_tf.py::TCNBlock`:\n",
            "\n",
            "```python\n",
            "class TCNBlock(layers.Layer):\n",
            "    def __init__(self, filters, kernel_size=5, dilation_rate=1, dropout=0.2):\n",
            "        super().__init__()\n",
            "        # Two causal dilated convolutions\n",
            "        self.conv1 = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, \n",
            "                                   padding='causal', activation='relu')\n",
            "        self.dropout1 = layers.Dropout(dropout)\n",
            "        \n",
            "        self.conv2 = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate,\n",
            "                                   padding='causal', activation='relu')\n",
            "        self.dropout2 = layers.Dropout(dropout)\n",
            "        \n",
            "        self.downsample = None  # Created in build() if needed\n",
            "        self.relu = layers.Activation('relu')\n",
            "    \n",
            "    def call(self, x, training=None):\n",
            "        residual = x\n",
            "        out = self.conv1(x)\n",
            "        out = self.dropout1(out, training=training)\n",
            "        out = self.conv2(out)\n",
            "        out = self.dropout2(out, training=training)\n",
            "        \n",
            "        if self.downsample is not None:\n",
            "            residual = self.downsample(residual)\n",
            "        \n",
            "        return self.relu(out + residual)\n",
            "```\n",
            "\n",
            "### 3.3 Design Choices\n",
            "\n",
            "**Activation**: ReLU (computational efficiency + non-saturation)\n",
            "\n",
            "**Dropout**: $p=0.2$ default (regularization without over-suppression)\n",
            "\n",
            "**Weight Init**: Orthogonal (stable gradient norms, He et al. 2015)\n",
            "\n",
            "**Padding**: 'causal' ensures no future leakage"
        ]
    })
    
    # Save updated notebook
    with open(r"c:\Users\Owner\new_project\adaptive_portfolio_rl\tcn_documentation\03_tcn_architectures_and_variants_EXPANDED.ipynb", 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✓ Added sections 2-3. Total cells: {len(notebook['cells'])}")
    return notebook

if __name__ == "__main__":
    nb = build_complete_tcn_docs()
