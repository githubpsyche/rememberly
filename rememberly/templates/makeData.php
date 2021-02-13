<?php

        if($_SERVER['REQUEST_METHOD'] == 'POST') {

                function get_data() {
                        $text = $_POST['textInput'];
                        return json_encode($text);
                }

                $name = "makeData";
                $file_name = $name . '.json';

                if(file_put_contents("$file_name", get_data())) {
                        echo $file_name .' file created';
                }
                else {
                        echo 'There is some error';
                }
        }

?>
