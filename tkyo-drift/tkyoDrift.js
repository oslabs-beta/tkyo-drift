#!/usr/bin/env node
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*::::-%@@#:..-:..+@@@@@@@@@%#++==+*%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%##########********#######%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@+:#@@@@@@=.-=+#++:..:=+*##*=..*%%@@@@%#=:@@@@@@@@@@%**@@@@@@@@@@@%#+=-::..::-==+**######%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%#:.-@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@@@@@@@@@@@@@@@@+%@@@@@@@@@@:.-:=#@@@@@@@@@@@@@%%=%@@@@@@%*.:-*%%%%*-*+=-:.:-=+*#%%######%%%%###***+++=====--------======+++++++****####%%%%@@%%#=--:+@@@@@@@@@@@@@@@@@@@@@@@=-@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@+*@@@@@@@@@@@@@@@@@@@@@%%%@@@@@@%-%%%+..=#%%#####*+=--=+**##%%%%%#*++=--::.........................................:%@%.....:...:::-=+*#%@@@@@@@@@**:@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@#%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%@@@@@%#%#:-%####%%#**+=-:.                                                                 -%#. :.....::::::::::::::::::-*.@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@.#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%@@%@@@%%=:%%*..                                                                              .%@:..:.....::::::::::::::::=-:@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@=-@@*@@@@@@@@@@@@@@@@@@@@@%*==-:.-#@@@@@@@@@@@@@@@@@@@@@%:#%+..            .************=:*****:.=****+-+****===*####=--+#%@@@@%#*::::...........%%:..-.   :@@@@@@@@@@@@@@@==-@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@@=--@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%+.*@@@@@@@@@@@@@@@@@@=-%%..       ......:@@@@@@@@@@@@%.%@@@@=+@@@@%:.*@@@@#:.#@@@@+.%@@@@@@@@@@@@:....     --.=.#@-..::    +@@@@@@@@@@@@@+=+@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@@#-%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#:@@@#@@@@@@@@@@@@%.*#+.:=++:.     .......:%@@@@*:...%@@@@%@@@@@-..-@@@@@=.*@@@@%.@@@@@=..@@@@@:           -=--:%%...--    .%@@@@@@@@@@@=-%@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@%.#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=@@@@-*@@@@@@@@@*-%%.:-:-=*@@%::... .+.  %@@@@#.#-.#@@@@@@@@@@@@*.-@@@@@@@@@@@=.*@@@@#..%@@@@**:          .=+=-.%@:..--.    =@@@@%%%%%@*=@@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@-+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%@@@@@@#-:-=+*+-=##:--:=-..:==%@%=-.....#@@@@%.   :@@@@@+=#@@@@%. .-@@@@@@@%:..=@@@@@. =@@@@%. ..:.         =+=:-@%...:=..+%%#****####%-%@@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@%.%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.+#+:-+=+:.   . =---%%+-.:@@@@@.   .@@@@@#:+@@@@@:    #@@@@%.....@@@@@@@@@@@@#.               :+++.=@%=:.=:=:+@@%########%.@@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@*:%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=-:----....:#@:##=::+++.       ...==.-@*@@@@@-... *@@@@-..%@@@@%. ..*@@@@#. ....=@@@@@@@@@+.                  -++-.*@*==..*:@@%%######%*+=@@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@*:%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@+*####*=:=*=-%*.*#=--+++.          ....+:.+@=-.  ..             ...##-.......... ....       ...........         .=-=.-@@*-+-=............=@@@@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@%.*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.%#####%@*::.-:*#-.=+==.                 .=..##:...          ........:##:........... .-=+++====------------:.   ......*@@=-==-.:+*#*:%%%#-:#@@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@=:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-:.:::...:=#:##=:-+=-.                   .....*@###########%@@#***%@###########@@#===+@%###########@@+*@%###########%@@@##############@%***-:@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@:=:.#@@@@@@@@@@@@@@@@@@@@%@@@@@@@@@@@@%#+:...+**#+ . ....:-==+++=---::::-==++**#%@+             :@@@@@-            -@@@@@#           +@@@@+           .@@%.             *@****+:.=@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@#.=++++.-#%@@@@@@@@@@@@@@@%-@@@@@@@@@@@@@@@*:%-::==***#%@@@@@@@@@@@@@@@@@@@@@@@@@@@@*    +@@@%.    *@@@@#.   %@@@.   .#@@@@@@@@:    #@@@@@@@%.   +@@@@@@@@@@@@@@=    :@@@@@@###%%*.:.:@@@@@@@@@@@@@@@@@@@        
@@@@%:. ..-+**-....=%%%@@@@@@@@%#:*@@@@@@@@@@@@@+.+-.++==-*##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%.   +@@@@@.   .@@@@@    -@@@#.  :@@@@@@@@@*    =@@@@@@@@.   :@@@@@@#*@@@#.%@:    @@@*++-#@@@@*.:..*@@@@@@@@@@@@@@@@@@        
@@@+-#@@@@@@@@@@@@+.=:..-+++=:.=.#@@@@@@@@@@@@@@#..#@%@-=*##%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    -@@@@@=   .@@@@@:            -#@@@@@@@%.   -@@@@@@@@-         #@@@@@@=%@-   .#%-#%%*+@@@@@+.:  -@@@@@@@@@@@@@@@@@@        
%-@@@@@@@@@@@@@@@@@+..+@=.*@@@@@.*%%@@@@%%@@%..*-=*@@#@:**%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@-   .@@@@@+    #@@@@-    -@@@-    =@@@@@@@@.   :@@@@@@@@=    +@@@@@@@@@@@%*@+    +@:.     .=@@@=.:  :@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@@@@@@@@*%@@@@@+*@@@@@#=..=*@@@.:**##*:%@@##-+*%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@*    -***+-    #@@@@+    %@@@@@.   =@@@@*++:    :++%@%%@*    *@#**=#%@@@@@@@%.   =@+..      ..:*-.:  :%@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@@@@@@@%*@@@@@@@*@@@@@@@@:-%%%:.#####+:%@@#=*++#%%%%%*%%@@@@@@@@@@@@@@@@@@@@@*            .-@@@@@%.   #@@@@@.    @@@@.          -@%#@@.   *@#**-%%*-...-@%    :@@:    ..  ....:.:  .%@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@@@@@@%@@@@@@@@@@@@@@@@@@@%.   -**###-:@@@%**=*==#%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@@@@@%%%%%@@@@@@@%%%@@@@@@@#=@@@@@@@@@@@@@@*:+@@@@@@@#++=:=++++++*@@@@@@@@*.    .=..  .:..:. .#@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@: .=-.=+%@%%-...-*#-:+++++++++++++====---:::...........................::::--------:::.::.-++++++++++++=.++++++++++++-:+++++++=.=#@@@@*+.    :.-.: ....:=*=#@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%.:=-:..=+%@%%-.-+++++:-++++++++++=:=++++++++++++-.:-:Authored By:                      ::.=+++++++++++++:-++++++++++++:-+++++-.=:=%@@@@=-     :.:.: .::.-**=*@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@=@@@@@@@@@@@@@@@@+..%@@.-=:-   =+#@@%=.-=++++=.=++++++++++:=+++++++++++++=..-:Tico, Milti, Anthony, Wing, Monique==++++++=======.-=====--:::::--=*##*-..:=@@@@%=:     .::.:  --.-**-*@@@@@@@@@@@@@@@@@        
@@@@@@@@@@*-@@@@@@@@@@@@@@@@@@@-=-:=....  =+*@@@+ .:-++++:.==++++++++=.==+++++++++++==...................::--==++*****+++======-----:::::---==+++*****######%%%#%%%%*@@@@%=.     .=:.: ..:::**-#@@@@@@@@@@@@@@@@@        
@@@@@@@@@@+:@@@@@@@@@@@@@@@@@@@@.:=.......-##@@@*...-**+++**##%%%%%%%%%%%%##**+++=======+++***###%%%%%%@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%#####***###%%%%@@@@@@@%%%%%%%%#@@@@%:....  .=... :.:-.**-#@@@@@@@@@@@@@@@@@        
@@@@@@@@@@#.+@@@@@@@@@@@@@@@@@@@.=-.-==....##@@@%**%%%%%%####*******####%%%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%#%%%%%%%@@@@*..:..  .-... :..- =*=%@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@#.:%@@@@@@@@@@@@@@@@%.=.=+==.:..=%@@@@%#@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%#####***********+++============+**##%%%%%%%##%@@@@@@@@@@@@@@@@@@@%%##**+*@%%@#*++=.-::..  .:... :..-.:*-###%%%%@@@@@@@@@@@        
@@@@@@@@@@@@@*...-#@@@@@@@@@@@@#:=:+==-.:...=%@@@@%%@@@@@@@@@@@@@@@@@@@@@%%@@@@%%%%%##**+===-----:::::........... ......:::--.    .  .%@#--:::---=--::::.-=++++:%%@@@@@#:.=.:..  ..... ..::.....................:        
@@@@@@@@@@@@@@@@@@@%.+@@@@@@@@@*-=-+=-=:..    -+#%%%++********########***%@-..  ....:===:....::::........:::::::::::::::..:===-   : .=.%@%########*+..    .=++=#@@@@@@@%.-=..    .. :.  -:....:::----==+*%@@@@@@@        
@@@@@@#@@@@@@@@@@@@@+:@@@@@@@@@*:====-=:..    .%@@@#=++++.....++++***####@%-..  : ..-==-:.........................:::-------==+:.::  :+-%@%########*..     =-=#@@@@@@@@-.=::.      .:  .::.#@@@@@@@@@@@@@@@@@@@@@        
@@@@@@+%@@@@@@@@@@@@@@*:-+%@@@@%:+++==-=..    .=@@@@#-++.      +########@@*-=.  :-:-===:::::::::::.:::::::::::::::.........:-==++====.:+=@@%#######%%=. .:-.+%@@@@@@@@#.:-..       .  .::.:++++++***#####%%%%%@@@        
@@@@@@@+*@@@@@@@@@@@@@@-@@@@@@@@.=*+====-.    ..#@@@@%=:-     .%#######%@@-+::=====++=:.......................:::-------::::.-==-:--.. =#=@@@%#%%@@@@@%=:*=#@@@@@@@@@%-.-..           ::.-@@@@@@@@@@@@@@@@@@@%%%%        
@@@@@@@@%:+@@@@@@@@@@@@.@@@@@@@@*-+*+*##*-..  ..=@@@@@@*=:=*#%@@@%%##%@@@#==. ..::-===:::::::::::::::::--------::::::::::::::::::::--==-:-%@@%%##%%%@%%@@@@@@@@@@@@@%+....           :..:@@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@%::*@@@@@@@@=.@@@@@@@@:-=*#####*:.....*@@@@@@@@@%####*++*#%@@@-.-++=---::::::..............:::---==+++++++++++*++=-:::-+#%%@@@@@@@@@@@@@@@@@@@@@@@@@@@%#+-.            .:. .#@@@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@@@@-++=+++=+-.#@@@@@@%.-+=*=+====-...:*%@@@@@@@@@@@@@@@@@@@@@@@%%%##**++====--------=================+++++***##%%@@@@@@@@@@@@@@@%%%%###***++==--::::..             ... ..-***###%%@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@@@@@@@#-++++=*=-=+.-%@@@@@:.-+===+-.   ..:***#%%%%%%%@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%##**++==--:::....                                                                                                         
@@@@@@@@@@@@@@@@@%+:--::=****=:..::-. ......       ...:::::..........................                                                                                                                           .        
@%%%####******+++++++++=============------:::::.............                                                   ...............................::::::::::::::::::::::------=====+++++++*******#######%%%%%%@@@@@@@        
@@@@@@@@@@@@@@@@@@%%%##############%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
import tkyoDriftSetTrainingHook from './util/tkyoDriftSetTrainingHook.js';
import printScalarCLI from './util/printScalarCLI.js';
import printLogCLI from './util/printLogCLI.js';
import tkyoDrift from './util/oneOffEmb.js';
import chalk from 'chalk';
import path from 'path';
import fs from 'fs';

// Get the commands from the CLI (the first 2 are not commands)
const [command, ...rest] = process.argv.slice(2);

// Only run if the command is a "tkyo" command
// if (process.argv[1] === new URL(import.meta.url).pathname) { // ! Alternative, ESM Based
if (process.argv[1].endsWith('tkyo')) {
  // switch case to determine which file to invoke
  switch (command) {
    // ? tkyo cos <number of days>
    case 'cos': {
      const dayArgument = rest[0] || '30';
      process.argv = ['node', 'printLogCLI.js', dayArgument];
      await printLogCLI(dayArgument);
      break;
    }

    // ? tkyo scalar
    case 'scalar': {
      await printScalarCLI();
      break;
    }

    // ? tkyo train <path to data> <column name> <ioType>
    case 'train': {
      const [pathToData, columnName, ioType] = rest;

      // Error handle when
      if (!pathToData || !columnName || !ioType) {
        console.error(
          chalk.blueBright(
            'Usage: tkyo train <path to data> <column name> <ioType>'
          )
        );
        process.exit(1);
      }

      // If someone calls the train command, we normalize the path.
      const normalizedPath = path.resolve(
        process.cwd(),
        pathToData.replace(/\\/g, '/')
      );

      // Error handle when the path does not exist.
      if (!fs.existsSync(normalizedPath)) {
        console.error(chalk.red(`The dataSetPath provided does not exist.`));
      }

      await tkyoDriftSetTrainingHook(normalizedPath, columnName, ioType);
      console.log(chalk.green("Job's done."));
      break;
    }

    // ? help commands
    default:
      console.log(
        chalk.gray(`
↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓    ↑↑↑     ↗↓↓↓↗     ↓↓↓         ↓↓↓    ↓↓↓↓↓↓↓↓↓↓↓↓↖
       ↑↑↑          ↑↑↑    ↗↑↑↑       ↑↑↑         ↑↑↑   ↑↑↑↑         ↖↑↑
      ↑↑↑          ↑↑↑   ↗↑↑↑        ↑↑↑         ↑↑↑   ↑↑↑           ↖↑↑
     ↑↑↑          ↑↑↑↑↑↑↑↘          ↑↑↑        ↑↑↑↑   ↑↑↑            ↖↑↑
    ↖↑↑         →↑↑    ↑↑↑↘         ↑↑↑↑↑↑↑↑↑↑↑↑↑    ←↑↑            ↑↑↑↗
    ↑↑↑         ↑↑↑     ↑↑↑↘             ↑↑↑         ↑↑↑           ↗↑↑↓
   ↑↑↑         ↑↑↑       ↑↑↑↘           ↑↑↑          ↑↑↑↑        ↗↑↑↑
  ↑↑↑         ↑↑↑         ↑↑↑↘         ↑↑↑            ↑↑↑↑↑↑↑↑↑↑↑↑↑↗

Usage:
  ${chalk.yellowBright('tkyo')} ${chalk.white('cos')} ${chalk.blueBright(
          '<number of days>'
        )}                         Show COS Drift logs for last N days
  ${chalk.yellowBright('tkyo')} ${chalk.white(
          'scalar'
        )}                                       Show scalar drift comparison
  ${chalk.yellowBright('tkyo')} ${chalk.white('train')} ${chalk.blueBright(
          '<path to data> <column name> <ioType>'
        )}  Embed dataset and update training baseline

Readme docs in the node package or at ${chalk.blueBright(
          'https://github.com/oslabs-beta/tkyo-drift'
        )}
      `)
      );
  }
}

export default tkyoDrift;
