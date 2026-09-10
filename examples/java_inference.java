// ONNX Runtime Java example. Rename to CaptchaPredictor.java before compiling.
import ai.onnxruntime.*;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.Map;
import javax.imageio.ImageIO;

class CaptchaPredictor implements AutoCloseable {
    private static final String CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    private final OrtEnvironment environment = OrtEnvironment.getEnvironment();
    private final OrtSession session;

    CaptchaPredictor(Path model) throws OrtException {
        session = environment.createSession(model.toString(), new OrtSession.SessionOptions());
    }

    private float[] preprocess(Path path) throws Exception {
        BufferedImage source = ImageIO.read(path.toFile());
        BufferedImage image = new BufferedImage(216, 96, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        graphics.drawImage(source, 0, 0, 216, 96, null);
        graphics.dispose();
        float[] values = new float[3 * 96 * 216];
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};
        for (int y = 0; y < 96; y++) for (int x = 0; x < 216; x++) {
            int rgb = image.getRGB(x, y);
            int offset = y * 216 + x;
            values[offset] = (((rgb >> 16) & 255) / 255f - mean[0]) / std[0];
            values[96 * 216 + offset] = (((rgb >> 8) & 255) / 255f - mean[1]) / std[1];
            values[2 * 96 * 216 + offset] = ((rgb & 255) / 255f - mean[2]) / std[2];
        }
        return values;
    }

    String predict(Path image) throws Exception {
        String input = session.getInputNames().iterator().next();
        try (OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(preprocess(image)), new long[]{1, 3, 96, 216});
             OrtSession.Result result = session.run(Map.of(input, tensor))) {
            float[][][] output = (float[][][]) result.get(0).getValue();
            StringBuilder text = new StringBuilder();
            int previous = -1;
            for (float[][] step : output) {
                int best = 0;
                for (int i = 1; i < step[0].length; i++) if (step[0][i] > step[0][best]) best = i;
                if (best == CHARS.length()) previous = -1;
                else if (best != previous) { text.append(CHARS.charAt(best)); previous = best; }
            }
            return text.toString();
        }
    }

    public void close() throws OrtException { session.close(); }
}
