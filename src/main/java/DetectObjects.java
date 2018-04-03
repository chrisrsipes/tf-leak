import java.util.List;
import java.util.Map;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import java.io.File;

public class DetectObjects {

  public final static String savedModelPath = "./src/main/resources/saved_model";

  public static void main(String[] args) throws Exception {
    int ch;
    List<Tensor<?>> outputs = null;

    try {

      while ((ch = System.in.read()) != -1) {
        try (SavedModelBundle model = SavedModelBundle.load(savedModelPath, "serve")) {

          try {
            outputs = model.session().runner().run();
          } catch (Exception e) {
          } finally {
            if (outputs != null) {
              for (Tensor<?> tensor : outputs) {
                tensor.close();
              }
            }
          }
        }
      }

    } catch (Exception e) {
    }
  }

}
