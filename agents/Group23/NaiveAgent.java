import java.net.*;
import java.util.ArrayList;
import java.util.Random;
import java.io.*;

class NaiveAgent{
    private BufferedReader in;

    private String colour = "R";
    private int turn = 0;
    private int boardSize = 11;

    public NaiveAgent(String agentColour, int gameBoardSize) {
        boardSize = gameBoardSize;
        colour = agentColour;
    }

    private String getMessage() throws IOException {
        return in.readLine();
    }

    private void sendMessage(String msg){
        System.out.print(msg+"\n");
        System.out.flush();
    }

    public void run(){
        in = new BufferedReader(new InputStreamReader(System.in));

        while (true){
            // receive messages
            try{
                String msg = getMessage();
                boolean res = interpretMessage(msg);
                if (res == false) break;
            } catch (IOException e){
                System.out.println("ERROR: Could not establish I/O.");
                return;
            }
        }
    }

    private boolean interpretMessage(String s){
        turn++;

        String[] msg = s.strip().split(";");
        String board = msg[2];
        switch (msg[0]){
            case "START":
                if (colour.equals("R")){
                    makeMove(board);
                }
                break;

            case "CHANGE":
                makeMove(board);
                break;

            case "SWAP":
                colour = opp(colour);
                makeMove(board);
            default:
                return false;
        }

        return true;
    }

    private void makeMove(String board){
        if (turn == 2 && new Random().nextInt(2) == 1){
            sendMessage("-1,-1");
            return;
        }

        String[] lines = board.split(",");
        ArrayList<int[]> choices = new ArrayList<int[]>();

        for (int i = 0; i < boardSize; i++)
            for (int j = 0; j < boardSize; j++)
                if (lines[i].charAt(j) == '0'){
                    int[] newElement = {i, j};
                    choices.add(newElement);
                }

        if (choices.size() > 0){
            int[] choice = choices.get(new Random().nextInt(choices.size()));
            String msg = "" + choice[0] + "," + choice[1];
            sendMessage(msg);
        }
    }

    public static String opp(String c){
        if (c.equals("R")) return "B";
        if (c.equals("B")) return "R";
        return "None";
    }


    public static void main(String args[]){
        String inputColour = args[0];
        String inputBoardSize = args[1];
        int parsedSize = Integer.parseInt(inputBoardSize);
        NaiveAgent agent = new NaiveAgent(inputColour, parsedSize);
        agent.run();
    }
}