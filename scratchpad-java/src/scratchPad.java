import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class scratchPad {
    public static void main(String[] args) {
        String regex = "(\\w+,)(\\d+,\\d+)";
        String input = "PhilipsLight,14,22";

        Pattern r = Pattern.compile(regex);
        Matcher m = r.matcher(input);

        if (m.find()) {
            System.out.println(m.group(2));
        }
    }
}
