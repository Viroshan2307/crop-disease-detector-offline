package com.example.executorchllamademo

import android.content.ContentResolver
import android.net.Uri
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import kotlin.math.exp

data class LeafResult(
    val label: String,
    val confidence: Float,
    val top3: List<Pair<String, Float>>
) {
    fun isSure(threshold: Float = LeafClassifier.THRESHOLD): Boolean = confidence >= threshold
}

class LeafClassifier(modelPath: String) {
    private val module: Module = Module.load(modelPath)

    fun classify(resolver: ContentResolver, uri: Uri): LeafResult {
        val img = ETImage(resolver, uri, SIZE)
        val bytes = img.bytes
        val plane = SIZE * SIZE
        require(bytes.size == 3 * plane) { "Could not read the photo" }

        val input = FloatArray(3 * plane)
        for (c in 0 until 3) {
            for (i in 0 until plane) {
                val v = (bytes[c * plane + i].toInt() and 0xFF) / 255.0f
                input[c * plane + i] = (v - MEAN[c]) / STD[c]
            }
        }
        val tensor = Tensor.fromBlob(input, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong()))
        val logits = module.forward(EValue.from(tensor))[0].toTensor().dataAsFloatArray

        val maxL = logits.max()
        val exps = FloatArray(logits.size) { exp(logits[it] - maxL) }
        val sum = exps.sum()
        val probs = FloatArray(exps.size) { exps[it] / sum }

        val order = probs.indices.sortedByDescending { probs[it] }
        val top3 = order.take(3).map { CLASSES[it] to probs[it] }
        return LeafResult(CLASSES[order[0]], probs[order[0]], top3)
    }

    companion object {
        const val SIZE = 224
        const val THRESHOLD = 0.6f
        val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
        val CLASSES = listOf(
            "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
            "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
            "Tomato___Leaf_Mold", "Tomato___Mosaic_virus", "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites", "Tomato___Target_Spot",
            "Tomato___Yellow_Leaf_Curl_Virus", "Tomato___healthy"
        )
    }
}